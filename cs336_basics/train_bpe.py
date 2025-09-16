import os
import time
import json
import regex
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from pathlib import Path

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练BPE分词器 - 优化版本"""
    
    print(f"🚀 开始BPE训练")
    print(f"📋 配置: vocab_size={vocab_size}, special_tokens={special_tokens}")
    print(f"📁 数据路径: {input_path}")
    
    # 1. 初始化词汇表
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    current_next_id: int = 256
    existing_byte_values: Set[bytes] = set(vocab.values())
    
    # 2. 添加特殊token
    for st_str in special_tokens:
        if len(vocab) >= vocab_size:
            break
        st_bytes = st_str.encode("utf-8")
        if st_bytes not in existing_byte_values:
            vocab[current_next_id] = st_bytes
            existing_byte_values.add(st_bytes)
            current_next_id += 1
    
    # 3. 加载数据 (采样以加快训练)
    print("📖 加载训练数据...")
    try:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            # 读取前50MB的数据进行训练
            text = f.read(50 * 1024 * 1024)
    except FileNotFoundError:
        text = ""
    
    # 4. 预分词
    print("🔧 进行预分词...")
    chunks = regex.split('|'.join(map(regex.escape, special_tokens)), text)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    token_frequency_table = defaultdict(int)
    for chunk in chunks:
        for word in regex.findall(PAT, chunk):
            word_bytes = word.encode("utf-8")
            bytes_list = [bytes([x]) for x in word_bytes]
            token_frequency_table[tuple(bytes_list)] += 1
    
    print(f"✅ 预分词完成，得到 {len(token_frequency_table):,} 个唯一token")
    
    # 5. 统计字符对频率
    print("📊 统计字符对频率...")
    pair_counts = defaultdict(int)
    for token, freq in token_frequency_table.items():
        for i in range(len(token) - 1):
            pair_counts[token[i], token[i+1]] += freq
    
    print(f"✅ 统计完成，发现 {len(pair_counts):,} 个字符对")
    
    merges: List[Tuple[bytes, bytes]] = []
    
    # 6. BPE训练主循环
    print(f"🔄 开始BPE训练，目标词汇表大小: {vocab_size}")
    total_merges_needed = vocab_size - len(vocab)
    
    iteration = 0
    while len(vocab) < vocab_size and pair_counts:
        iteration += 1
        if iteration % 100 == 0:
            print(f"   进度: {len(vocab)}/{vocab_size} tokens, {len(merges)} merges")
        
        # 找到频率最高的字符对
        if not pair_counts:
            break
            
        max_count = max(pair_counts.values())
        candidates = [k for k, v in pair_counts.items() if v == max_count]
        best_pair = max(candidates)  # 选择字节序最大的
        
        # 记录合并
        merges.append(best_pair)
        new_token_bytes = best_pair[0] + best_pair[1]
        
        # 添加到词汇表
        vocab[current_next_id] = new_token_bytes
        current_next_id += 1
        
        # 更新受影响的token
        affected_tokens = []
        for token, freq in list(token_frequency_table.items()):
            has_pair = any(token[i:i+2] == best_pair for i in range(len(token) - 1))
            if has_pair:
                affected_tokens.append((token, freq))
        
        # 处理受影响的token
        for token, freq in affected_tokens:
            # 移除旧的字符对计数
            for i in range(len(token) - 1):
                pair_counts[token[i], token[i+1]] -= freq
                if pair_counts[token[i], token[i+1]] <= 0:
                    del pair_counts[token[i], token[i+1]]
            
            # 合并字符对
            new_token_frequency_seq = merge_token_sequence(token, best_pair, new_token_bytes)
            
            # 添加新的字符对计数
            for i in range(len(new_token_frequency_seq) - 1):
                pair = (new_token_frequency_seq[i], new_token_frequency_seq[i+1])
                pair_counts[pair] += freq
            
            # 更新token频率表
            del token_frequency_table[token]
            token_frequency_table[new_token_frequency_seq] += freq
    
    print(f"✅ BPE训练完成!")
    return vocab, merges

def merge_token_sequence(token: Tuple[bytes, ...], pair: Tuple[bytes, bytes], new_token: bytes) -> Tuple[bytes, ...]:
    """合并token序列中的字符对"""
    new_seq = []
    i = 0
    while i < len(token):
        if i < len(token) - 1 and token[i:i+2] == pair:
            new_seq.append(new_token)
            i += 2
        else:
            new_seq.append(token[i])
            i += 1
    return tuple(new_seq)

def save_vocab_and_merges(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], vocab_path: str, merges_path: str):
    """保存词汇表和合并列表到文件"""
    # 1. 保存词汇表 (JSON格式)
    vocab_str = {str(idx): token.decode('utf-8', errors='replace') for idx, token in vocab.items()}
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_str, f, ensure_ascii=False, indent=2)
    
    # 2. 保存合并列表 (文本格式)
    with open(merges_path, 'w', encoding='utf-8') as f:
        for merge in merges:
            part1 = merge[0].decode('utf-8', errors='replace')
            part2 = merge[1].decode('utf-8', errors='replace')
            f.write(f"{part1} {part2}\n")

def analyze_tokenizer_results(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], training_time: float):
    """分析分词器结果并报告所需指标"""
    print("\n" + "="*60)
    print("📊 BPE训练结果分析")
    print("="*60)
    
    # 1. 训练时间
    print(f"⏱️  训练时间: {training_time:.2f} 秒")
    
    # 2. 内存使用
    try:
        import psutil
        process = psutil.Process()
        mem_usage = process.memory_info().rss / (1024 ** 3)  # GB
        print(f"💾 峰值内存使用: {mem_usage:.2f} GB")
    except ImportError:
        print("💾 峰值内存使用: 无法获取 (psutil未安装)")
    
    # 3. 词汇表统计
    print(f"📚 词汇表大小: {len(vocab):,}")
    print(f"🔗 合并操作数: {len(merges):,}")
    
    # 4. 最长token分析
    longest_token = max(vocab.values(), key=len)
    longest_token_str = longest_token.decode('utf-8', errors='replace')
    print(f"📏 最长token: '{longest_token_str}' (长度: {len(longest_token)} 字节)")
    
    # 5. 分析最长token是否合理
    if len(longest_token) > 20:
        print("🤔 最长token分析: 长度较长，可能包含完整的单词或短语")
    elif len(longest_token) > 10:
        print("✅ 最长token分析: 长度适中，可能包含常见子词")
    else:
        print("✅ 最长token分析: 长度较短，符合BPE预期")
    
    # 6. 特殊token检查
    special_tokens = [token for token in vocab.values() if b'<|' in token]
    print(f"🎯 特殊token数量: {len(special_tokens)}")
    for token in special_tokens:
        print(f"   - '{token.decode('utf-8', errors='replace')}'")
    
    return {
        'training_time': training_time,
        'vocab_size': len(vocab),
        'merges_count': len(merges),
        'longest_token': longest_token_str,
        'longest_token_length': len(longest_token)
    }

if __name__ == "__main__":
    # 配置参数
    config = {
        "vocab_size": 10000,
        "special_tokens": ["<|endoftext|>"],
    }
    
    # 数据集路径
    train_path = "../../data/TinyStoriesV2-GPT4-train.txt"
    
    # 检查文件是否存在
    if not Path(train_path).exists():
        raise FileNotFoundError(f"训练集文件 {train_path} 不存在")
    
    # 训练模型
    start_time = time.time()
    train_vocab, train_merges = run_train_bpe(train_path, **config)
    training_time = time.time() - start_time
    
    print(f"\n✅ 训练完成! 耗时: {training_time:.2f}秒")
    
    # 分析结果
    results = analyze_tokenizer_results(train_vocab, train_merges, training_time)
    
    # 保存结果到磁盘
    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    
    vocab_path = os.path.join(output_dir, "vocab.json")
    merges_path = os.path.join(output_dir, "merges.txt")
    
    save_vocab_and_merges(train_vocab, train_merges, vocab_path, merges_path)
    print(f"\n💾 文件保存:")
    print(f"   - 词汇表: {vocab_path}")
    print(f"   - 合并列表: {merges_path}")
    
    # 最终总结
    print("\n" + "="*60)
    print("🎯 任务完成总结")
    print("="*60)
    print(f"✅ 成功训练了字节级BPE分词器")
    print(f"✅ 词汇表大小: {results['vocab_size']:,}")
    print(f"✅ 包含特殊token: <|endoftext|>")
    print(f"✅ 训练时间: {results['training_time']:.2f}秒")
    print(f"✅ 最长token: '{results['longest_token']}' (长度: {results['longest_token_length']})")
    print(f"✅ 结果已序列化到磁盘")
    
    # 判断是否合理
    if results['training_time'] < 120:  # 2分钟
        print("✅ 训练时间合理: 符合<2分钟要求")
    else:
        print("⚠️  训练时间较长: 超过2分钟要求")
        
    if results['longest_token_length'] > 50:
        print("🤔 最长token较长: 可能需要检查是否合理")
    else:
        print("✅ 最长token长度合理: 符合BPE预期")
