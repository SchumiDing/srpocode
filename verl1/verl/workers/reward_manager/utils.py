# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for reward managers supporting multiple model output formats."""

from typing import Optional, Tuple
import re
from sympy import simplify, parse_expr
from sympy.parsing.latex import parse_latex


def compare_latex_expressions(expr1: str, expr2: str) -> bool:
    """Compare two LaTeX expressions to check if they are mathematically equivalent.
    
    Args:
        expr1: First expression string
        expr2: Second expression string
        
    Returns:
        True if expressions are equivalent, False otherwise
    """
    if expr1.strip() == expr2.strip():
        return True
    try:
        if int(expr1) == int(expr2):
            return True
    except:
        pass
    try:
        def normalize(expr):
            expr = re.sub(r'\s+', '', expr)
            return expr.replace(' ', '')
        
        if normalize(expr1) == normalize(expr2):
            return True
        
        try:
            sympy1 = parse_latex(expr1)
            sympy2 = parse_latex(expr2)
            return simplify(sympy1 - sympy2) == 0
        except:
            return False
            
    except Exception as e:
        return False


def parse_llama3_conversation(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse Llama3 chat template applied text, extract system and user content.
    
    Args:
        text: Text with Llama3 chat template applied
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (system_content, user_content)
    """
    pattern = r'<\|start_header_id\|>([^<]+)<\|end_header_id\|>\s*\n{0,2}(.*?)\s*<\|eot_id\|>'
    
    matches = re.findall(pattern, text, re.DOTALL)
    
    system_content = None
    user_content = None
    
    for role, content in matches:
        role = role.strip()
        content = content.strip()
        
        if role == "system" and system_content is None:
            system_content = content
        elif role == "user" and user_content is None:
            user_content = content
    
    return (system_content, user_content)


def parse_qwen_conversation(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse Qwen chat template applied text, extract system and user content.
    
    Args:
        text: Text with Qwen chat template applied
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (system_content, user_content)
    """
    system_content = None
    user_content = None
    
    # Match system message
    system_pattern = r'<\|im_start\|>system\s*\n(.*?)<\|im_end\|>'
    system_matches = re.findall(system_pattern, text, re.DOTALL)
    
    if system_matches:
        # Take the first system message
        system_content = system_matches[0].strip()
    
    # Match user message
    user_pattern = r'<\|im_start\|>user\s*\n(.*?)<\|im_end\|>'
    user_matches = re.findall(user_pattern, text, re.DOTALL)
    
    if user_matches:
        # Take the first user message
        user_content = user_matches[0].strip()
    
    return (system_content, user_content)


def parse_deepseek_conversation(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse DeepSeek chat template applied text, extract system and user content.
    
    DeepSeek chat template format:
    - System prompt: directly after bos_token, before first <｜tool▁calls▁begin｜>
    - User message: <｜tool▁calls▁begin｜> + content
    - Assistant message: <｜Assistant｜> + content + 
    
    Args:
        text: Text with DeepSeek chat template applied
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (system_content, user_content)
    """
    system_content = None
    user_content = None
    
    # DeepSeek uses special markers
    # Extract system content: content before first <｜tool▁calls▁begin｜> (remove possible bos_token)
    user_tag = '<｜tool▁calls▁begin｜>'
    assistant_tag = '<｜Assistant｜>'
    
    # Find the position of first User marker
    first_user_pos = text.find(user_tag)
    
    if first_user_pos > 0:
        # system content is from bos_token to first <｜tool▁calls▁begin｜>
        system_content = text[:first_user_pos].strip()
        # Remove possible bos_token
        bos_pattern = r'^'
        system_content = re.sub(bos_pattern, '', system_content).strip()
        if not system_content:
            system_content = None
    
    # Extract user content: <｜tool▁calls▁begin｜> to next marker
    if first_user_pos != -1:
        # Start from after first User marker
        after_user_tag = first_user_pos + len(user_tag)
        remaining_text = text[after_user_tag:]
        
        # Find next marker position (could be Assistant, tool-related markers, etc.)
        # Possible end markers
        end_markers = [
            '<｜Assistant｜>',
            '<｜tool▁calls▁begin｜>',
            '<｜tool▁call▁end｜>',
            '',
            '<｜tool▁calls▁begin｜>',  # If there are multiple turns
        ]
        
        # Find the nearest end marker
        end_pos = len(remaining_text)
        for marker in end_markers:
            pos = remaining_text.find(marker)
            if pos != -1 and pos < end_pos:
                end_pos = pos
        
        user_content = remaining_text[:end_pos].strip()
        if not user_content:
            user_content = None
    
    return (system_content, user_content)
