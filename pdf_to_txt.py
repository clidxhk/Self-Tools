import os
import re
from typing import List
import PyPDF2
from openai import OpenAI
import tiktoken

class PDFTextExtractor:
    def __init__(self, openai_api_key: str, model: str = "gpt-5-mini", base_url: str = None):
        """
        初始化PDF文本提取器
        
        Args:
            openai_api_key: OpenAI API密钥
            model: 使用的LLM模型，默认为qwen3-4b
            base_url: 自定义API base URL（可选）
        """
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=base_url if base_url else "https://api.openai.com/v1"  # 默认使用OpenAI官方URL
        )
        self.model = model
        
        # 处理不同模型的token计算
        # try:
        #     # 尝试使用tiktoken自动获取编码器
        #     self.encoding = tiktoken.encoding_for_model(model)
        # except KeyError:
        #     # 如果无法自动识别，使用默认的cl100k_base编码器（适用于大多数模型）
        #     print(f"警告: 无法自动识别模型 {model} 的编码器，使用默认编码器 cl100k_base")
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def remove_references_section(self, text: str) -> str:
        """
        预处理文本，检测并删除"References"之后的所有内容
        
        Args:
            text: 原始文本
            
        Returns:
            删除参考文献部分后的文本
        """
        # 直接查找REFERENCES字符串的位置（不区分大小写）
        references_pos = re.search(r'REFERENCES', text, re.IGNORECASE)
        
        # 如果找到REFERENCES，删除它及其后的所有内容
        if references_pos:
            text_before_refs = text[:references_pos.start()]
            removed_length = len(text) - len(text_before_refs)
            print(f"检测到参考文献部分，已删除 {removed_length} 个字符")
            return text_before_refs
        
        # 如果没有找到参考文献部分，返回原文本
        print("未检测到参考文献部分")
        return text
    
    def pdf_to_text(self, pdf_path: str) -> str:
        """
        将PDF文件转换为纯文本
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的文本内容
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                    
            print(f"成功从PDF中提取了 {len(text)} 个字符的文本")
            return text
        except Exception as e:
            print(f"PDF转文本时出错: {e}")
            return ""
    
    def split_text_into_chunks(self, text: str, max_tokens: int = 8192) -> List[str]:
        """
        将文本分割为多个块,每个块不超过指定的token数量
        
        Args:
            text: 要分割的文本
            max_tokens: 每个块的最大token数量
            
        Returns:
            文本块列表
        """
        # 首先按段落分割
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # 如果当前段落为空,跳过
            if not paragraph.strip():
                continue
            
            # 检查单个段落是否超过限制
            paragraph_tokens = len(self.encoding.encode(paragraph))
            if paragraph_tokens > max_tokens:
                # 如果段落本身超过限制,需要先保存当前块,然后分割这个段落
                if current_chunk:
                    chunks.append(current_chunk)
                    print(f"添加块 {len(chunks)}: {len(current_chunk)} 字符")
                    current_chunk = ""
                
                # 按句子分割超大段落
                sentences = re.split(r'([.!?。!?])', paragraph)
                # 重新组合句子和标点
                sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''])]
                
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                        
                    test_chunk = current_chunk + sentence if current_chunk else sentence
                    token_count = len(self.encoding.encode(test_chunk))
                    
                    if token_count <= max_tokens:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                            print(f"添加块 {len(chunks)}: {len(current_chunk)} 字符, {len(self.encoding.encode(current_chunk))} tokens")
                        current_chunk = sentence
                continue
                
            # 计算添加这个段落后是否会超过token限制
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            token_count = len(self.encoding.encode(test_chunk))
            
            if token_count <= max_tokens:
                # 如果不超过限制,添加到当前块
                current_chunk = test_chunk
            else:
                # 如果超过限制,保存当前块并开始新块
                if current_chunk:
                    chunks.append(current_chunk)
                    print(f"添加块 {len(chunks)}: {len(current_chunk)} 字符, {len(self.encoding.encode(current_chunk))} tokens")
                current_chunk = paragraph
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
            print(f"添加最后一块 {len(chunks)}: {len(current_chunk)} 字符, {len(self.encoding.encode(current_chunk))} tokens")
            
        print(f"文本已分割为 {len(chunks)} 个块")
        # 打印每个块的token数用于调试
        for i, chunk in enumerate(chunks):
            token_count = len(self.encoding.encode(chunk))
            print(f"块 {i+1}: {len(chunk)} 字符, {token_count} tokens")
        return chunks
    
    def clean_text_chunk_with_llm(self, chunk: str) -> str:
        """
        使用LLM清理文本块，提取正文内容
        
        Args:
            chunk: 原始文本块
            
        Returns:
            清理后的文本
        """
        prompt = """
        请从以下文本中提取正文内容，删除页眉、页脚、页码、目录、参考文献等非正文内容。
        保留段落结构和重要信息，使文本更加连贯和可读。
        
        文本内容:
        {text}
        
        请只返回清理后的正文内容，不要添加任何解释或说明。
        """.format(text=chunk)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional text editing assistant, skilled at extracting the main content from academic documents, removing non-text elements such as headers, footers, page numbers, table of contents, references, and author information, figure captions,and formatting the text without line breaks. If there are no spaces between the words, add spaces."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            cleaned_text = response.choices[0].message.content
            return cleaned_text.strip()
        except Exception as e:
            print(f"LLM清理文本时出错: {e}")
            return chunk
    
    def merge_chunks_with_overlap(self, chunks: List[str], overlap_chars: int = 100) -> str:
        """
        合并文本块，确保所有清理后的内容都被保留，不添加换行符
        """
        if not chunks:
            return ""
            
        if len(chunks) == 1:
            return chunks[0]
        
        result = chunks[0]  # 保留第一个清理后的块
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            
            # 简单的重叠检测和处理
            overlap_found = False
            max_check_len = min(overlap_chars, len(result), len(current_chunk))
            
            # 从最长的可能重叠开始检查
            for check_len in range(max_check_len, 0, -1):
                if result[-check_len:] == current_chunk[:check_len]:
                    # 找到重叠，只添加非重叠部分
                    result = result + current_chunk[check_len:]
                    overlap_found = True
                    print(f"块 {i+1}: 找到重叠 {check_len} 字符")
                    break
            
            if not overlap_found:
                # 没有重叠，直接连接（不添加换行符）
                result = result + current_chunk
                print(f"块 {i+1}: 无重叠，直接连接")
        
        return result
    
    def _find_best_overlap(self, text1: str, text2: str) -> str:
        """
        在两个文本片段之间找到最佳重叠部分
        
        Args:
            text1: 第一个文本片段
            text2: 第二个文本片段
            
        Returns:
            重叠部分的文本，如果没有重叠则返回空字符串
        """
        max_overlap = min(len(text1), len(text2))
        best_overlap = ""
        
        # 从最长的可能重叠开始，逐渐减小
        for overlap_size in range(max_overlap, 10, -1):  # 最小重叠10个字符
            end_part = text1[-overlap_size:]
            start_part = text2[:overlap_size]
            
            if end_part == start_part:
                best_overlap = start_part
                break
                
        return best_overlap
    
    def extract_text_from_pdf(self, pdf_path: str, output_path: str = None, max_tokens=4096) -> str:
        """
        从PDF提取并清理文本的完整流程
        
        Args:
            pdf_path: PDF文件路径
            output_path: 输出文件路径（可选）
            
        Returns:
            提取并清理后的文本
        """
        print("开始PDF文本提取流程...")
        
        # 1. PDF转换为TXT
        print("步骤1: 将PDF转换为文本...")
        raw_text = self.pdf_to_text(pdf_path)
        if not raw_text:
            print("PDF转文本失败，终止流程")
            return ""

        print(f"原始文本长度: {len(raw_text)} 字符")
        print(f"原始文本前100字符: {raw_text[:100]}")

        # 保存原始文本（用于检查）
        raw_text_output_path = pdf_path.replace('.pdf', '_raw.txt')
        try:
            with open(raw_text_output_path, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            print(f"原始文本已保存到: {raw_text_output_path}")
        except Exception as e:
            print(f"保存原始文本时出错: {e}")
        
        # 2. 预处理：删除参考文献部分
        print("步骤2: 预处理文本，删除参考文献部分...")
        processed_text = self.remove_references_section(raw_text)
        print(f"预处理后文本长度: {len(processed_text)} 字符")
        
        # 3. 把PDF拆分为若干块
        print("步骤3: 将文本分割为块...")
        chunks = self.split_text_into_chunks(processed_text, max_tokens=max_tokens)
        
        # 4. 使用LLM对每一块进行清理
        print("步骤4: 使用LLM清理每个文本块...")
        cleaned_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"正在处理块 {i+1}/{len(chunks)}，原始长度: {len(chunk)} 字符...")
            cleaned_chunk = self.clean_text_chunk_with_llm(chunk)
            print(f"块 {i+1} 清理后长度: {len(cleaned_chunk)} 字符")
            print(f"块 {i+1} 前50字符: {cleaned_chunk[:50]}")
            cleaned_chunks.append(cleaned_chunk)
        
        # 5. 按照顺序合并块，避免重复
        print("步骤5: 合并清理后的文本块...")
        final_text = self.merge_chunks_with_overlap(cleaned_chunks)
        
        print(f"最终文本长度: {len(final_text)} 字符")
        print(f"最终文本前100字符: {final_text[:100]}")
        
        # 保存结果（如果指定了输出路径）
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(final_text)
                print(f"结果已保存到: {output_path}")
            except Exception as e:
                print(f"保存文件时出错: {e}")
        
        print("PDF文本提取流程完成!")
        return final_text

    def process_folder(self, input_folder: str, output_folder: str = None, max_tokens=4096):
        """
        处理整个文件夹中的PDF文件
        
        Args:
            input_folder: 输入文件夹路径，包含PDF文件
            output_folder: 输出文件夹路径（可选），如果未指定则使用输入文件夹
        """
        # 设置输出文件夹
        if output_folder is None:
            output_folder = input_folder
        
        # 创建输出文件夹（如果不存在）
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取所有PDF文件
        pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"在文件夹 {input_folder} 中未找到PDF文件")
            return
        
        print(f"找到 {len(pdf_files)} 个PDF文件:")
        for pdf_file in pdf_files:
            print(f"  - {pdf_file}")
        
        # 处理每个PDF文件
        successful_count = 0
        failed_count = 0
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_folder, pdf_file)
            # 生成输出文件名（替换扩展名为.txt）
            output_filename = pdf_file.replace('.pdf', '.txt')
            output_path = os.path.join(output_folder, output_filename)
            
            try:
                print(f"\n{'='*60}")
                print(f"正在处理第 {successful_count + failed_count + 1}/{len(pdf_files)} 个文件: {pdf_file}")
                print(f"{'='*60}")
                
                extracted_text = self.extract_text_from_pdf(pdf_path, output_path, max_tokens=max_tokens)
                
                if extracted_text:
                    successful_count += 1
                    print(f"✓ 成功处理: {pdf_file}")
                else:
                    failed_count += 1
                    print(f"✗ 处理失败: {pdf_file}")
                    
            except Exception as e:
                failed_count += 1
                print(f"✗ 处理 {pdf_file} 时出错: {e}")
        
        print(f"\n{'='*60}")
        print("批量处理完成!")
        print(f"成功处理: {successful_count} 个文件")
        print(f"处理失败: {failed_count} 个文件")
        print(f"总计: {len(pdf_files)} 个文件")
        print(f"{'='*60}")


# 使用示例
if __name__ == "__main__":
    # 配置参数
    OPENAI_API_KEY = ""  # 替换为您的OpenAI API密钥
    BASE_URL = ""  # 替换为您的自定义API base URL

    
    # 创建提取器实例
    extractor = PDFTextExtractor(
        openai_api_key=OPENAI_API_KEY,
        model="",
        base_url=BASE_URL
    )
    
    # PDF_FILE_PATH = "D:/Code/pdf_to_text/10.1021_jacs.9b05329.pdf"  # 替换为您的PDF文件路径
    # OUTPUT_FILE_PATH = "D:/Code/pdf_to_text/10.1021_jacs.9b05329.txt"  # 输出文件路径

    # 执行提取
    # extracted_text = extractor.extract_text_from_pdf(PDF_FILE_PATH, OUTPUT_FILE_PATH)
    
    # 打印部分结果
    # if extracted_text:
    #     print("\n提取的文本预览:")
    #     print("=" * 50)
    #     print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)

    INPUT_FOLDER = "F:/PDF/Convert"  # 替换为您的输入文件夹路径
    OUTPUT_FOLDER = "F:/PDF/Convert"  # 替

    extractor.process_folder(INPUT_FOLDER, OUTPUT_FOLDER, max_tokens=2048)