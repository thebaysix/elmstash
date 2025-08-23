"""
Pattern Analysis - Objective pattern detection in model interactions.

This module detects statistical patterns without making judgments
about whether patterns are good or bad.
"""

from typing import Dict, List, Any, Tuple
from collections import Counter
import re


class PatternAnalyzer:
    """Detects objective patterns in model interaction data."""
    
    def __init__(self):
        self.patterns_cache = {}
    
    def detect_repetition_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """
        Detect repetition patterns in text outputs.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Dictionary with repetition pattern statistics
        """
        if not texts:
            return {'no_data': True}
        
        # Exact string repetitions
        text_counts = Counter(texts)
        repeated_texts = {text: count for text, count in text_counts.items() if count > 1}
        
        # Phrase repetitions within texts
        phrase_repetitions = self._detect_phrase_repetitions(texts)
        
        # Pattern repetitions (e.g., "I can help", "I can assist")
        pattern_repetitions = self._detect_pattern_repetitions(texts)
        
        return {
            'no_data': False,
            'exact_repetitions': {
                'count': len(repeated_texts),
                'examples': list(repeated_texts.items())[:5],  # Top 5 examples
                'max_repetition': max(text_counts.values()) if text_counts else 0
            },
            'phrase_repetitions': phrase_repetitions,
            'pattern_repetitions': pattern_repetitions,
            'uniqueness_ratio': len(set(texts)) / len(texts) if texts else 0
        }
    
    def detect_length_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """
        Detect patterns in text lengths.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Dictionary with length pattern statistics
        """
        if not texts:
            return {'no_data': True}
        
        lengths = [len(text) for text in texts]
        
        # Length distribution
        length_counts = Counter(lengths)
        
        # Detect common length ranges
        length_ranges = self._categorize_lengths(lengths)
        
        # Detect length trends
        length_trends = self._detect_length_trends(lengths)
        
        return {
            'no_data': False,
            'length_distribution': {
                'unique_lengths': len(length_counts),
                'most_common_length': length_counts.most_common(1)[0] if length_counts else None,
                'length_variance': self._calculate_variance(lengths)
            },
            'length_ranges': length_ranges,
            'length_trends': length_trends
        }
    
    def detect_structural_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """
        Detect structural patterns in texts (punctuation, formatting, etc.).
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Dictionary with structural pattern statistics
        """
        if not texts:
            return {'no_data': True}
        
        # Punctuation patterns
        punctuation_patterns = self._analyze_punctuation_patterns(texts)
        
        # Capitalization patterns
        capitalization_patterns = self._analyze_capitalization_patterns(texts)
        
        # Sentence structure patterns
        sentence_patterns = self._analyze_sentence_patterns(texts)
        
        return {
            'no_data': False,
            'punctuation_patterns': punctuation_patterns,
            'capitalization_patterns': capitalization_patterns,
            'sentence_patterns': sentence_patterns
        }
    
    def detect_content_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """
        Detect content-based patterns (word usage, topics, etc.).
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Dictionary with content pattern statistics
        """
        if not texts:
            return {'no_data': True}
        
        # Word frequency patterns
        word_patterns = self._analyze_word_patterns(texts)
        
        # Starting phrase patterns
        start_patterns = self._analyze_start_patterns(texts)
        
        # Ending phrase patterns
        end_patterns = self._analyze_end_patterns(texts)
        
        return {
            'no_data': False,
            'word_patterns': word_patterns,
            'start_patterns': start_patterns,
            'end_patterns': end_patterns
        }
    
    # Private helper methods
    
    def _detect_phrase_repetitions(self, texts: List[str]) -> Dict[str, Any]:
        """Detect repeated phrases within and across texts."""
        all_phrases = []
        
        for text in texts:
            # Extract phrases (simple n-gram approach)
            words = text.lower().split()
            for i in range(len(words) - 2):  # 3-word phrases
                phrase = ' '.join(words[i:i+3])
                all_phrases.append(phrase)
        
        phrase_counts = Counter(all_phrases)
        repeated_phrases = {phrase: count for phrase, count in phrase_counts.items() if count > 1}
        
        return {
            'total_phrases': len(all_phrases),
            'unique_phrases': len(phrase_counts),
            'repeated_phrases': len(repeated_phrases),
            'most_repeated': phrase_counts.most_common(5) if phrase_counts else []
        }
    
    def _detect_pattern_repetitions(self, texts: List[str]) -> Dict[str, Any]:
        """Detect similar patterns using simple regex matching."""
        # Common patterns to look for
        patterns = {
            'i_can_pattern': r'\bi can\b',
            'question_pattern': r'\?',
            'exclamation_pattern': r'!',
            'greeting_pattern': r'\b(hello|hi|hey)\b',
            'apology_pattern': r'\b(sorry|apologize)\b'
        }
        
        pattern_matches = {}
        for pattern_name, pattern_regex in patterns.items():
            matches = []
            for text in texts:
                matches.extend(re.findall(pattern_regex, text.lower()))
            pattern_matches[pattern_name] = len(matches)
        
        return pattern_matches
    
    def _categorize_lengths(self, lengths: List[int]) -> Dict[str, int]:
        """Categorize lengths into ranges."""
        categories = {
            'very_short': 0,    # 0-50 chars
            'short': 0,         # 51-150 chars
            'medium': 0,        # 151-500 chars
            'long': 0,          # 501-1000 chars
            'very_long': 0      # 1000+ chars
        }
        
        for length in lengths:
            if length <= 50:
                categories['very_short'] += 1
            elif length <= 150:
                categories['short'] += 1
            elif length <= 500:
                categories['medium'] += 1
            elif length <= 1000:
                categories['long'] += 1
            else:
                categories['very_long'] += 1
        
        return categories
    
    def _detect_length_trends(self, lengths: List[int]) -> Dict[str, Any]:
        """Detect trends in length over sequence."""
        if len(lengths) < 3:
            return {'insufficient_data': True}
        
        increasing = sum(1 for i in range(1, len(lengths)) if lengths[i] > lengths[i-1])
        decreasing = sum(1 for i in range(1, len(lengths)) if lengths[i] < lengths[i-1])
        stable = len(lengths) - 1 - increasing - decreasing
        
        return {
            'insufficient_data': False,
            'increasing_steps': increasing,
            'decreasing_steps': decreasing,
            'stable_steps': stable,
            'overall_trend': 'increasing' if increasing > decreasing else 'decreasing' if decreasing > increasing else 'stable'
        }
    
    def _analyze_punctuation_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze punctuation usage patterns."""
        punctuation_counts = Counter()
        
        for text in texts:
            for char in text:
                if char in '.,!?;:':
                    punctuation_counts[char] += 1
        
        return {
            'total_punctuation': sum(punctuation_counts.values()),
            'punctuation_distribution': dict(punctuation_counts.most_common()),
            'texts_with_questions': sum(1 for text in texts if '?' in text),
            'texts_with_exclamations': sum(1 for text in texts if '!' in text)
        }
    
    def _analyze_capitalization_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze capitalization patterns."""
        patterns = {
            'all_caps_words': 0,
            'title_case_starts': 0,
            'lowercase_starts': 0,
            'mixed_case_unusual': 0
        }
        
        for text in texts:
            words = text.split()
            if not words:
                continue
                
            # Check first word capitalization
            first_word = words[0]
            if first_word.isupper():
                patterns['all_caps_words'] += 1
            elif first_word[0].isupper():
                patterns['title_case_starts'] += 1
            elif first_word[0].islower():
                patterns['lowercase_starts'] += 1
            
            # Count all-caps words
            for word in words:
                if len(word) > 1 and word.isupper():
                    patterns['all_caps_words'] += 1
        
        return patterns
    
    def _analyze_sentence_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze sentence structure patterns."""
        sentence_counts = []
        
        for text in texts:
            # Simple sentence counting (by periods, exclamations, questions)
            sentence_endings = text.count('.') + text.count('!') + text.count('?')
            sentence_counts.append(max(1, sentence_endings))  # At least 1 sentence
        
        return {
            'avg_sentences_per_text': sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0,
            'single_sentence_texts': sum(1 for count in sentence_counts if count == 1),
            'multi_sentence_texts': sum(1 for count in sentence_counts if count > 1),
            'max_sentences': max(sentence_counts) if sentence_counts else 0
        }
    
    def _analyze_word_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze word usage patterns."""
        all_words = []
        
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        
        return {
            'total_words': len(all_words),
            'unique_words': len(word_counts),
            'vocabulary_richness': len(word_counts) / len(all_words) if all_words else 0,
            'most_common_words': word_counts.most_common(10),
            'single_use_words': sum(1 for count in word_counts.values() if count == 1)
        }
    
    def _analyze_start_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze how texts typically start."""
        start_words = []
        start_phrases = []
        
        for text in texts:
            words = text.strip().split()
            if words:
                start_words.append(words[0].lower())
                if len(words) >= 2:
                    start_phrases.append(f"{words[0]} {words[1]}".lower())
        
        return {
            'start_word_distribution': Counter(start_words).most_common(10),
            'start_phrase_distribution': Counter(start_phrases).most_common(10),
            'unique_starts': len(set(start_words))
        }
    
    def _analyze_end_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze how texts typically end."""
        end_chars = []
        end_words = []
        
        for text in texts:
            text = text.strip()
            if text:
                end_chars.append(text[-1])
                words = text.split()
                if words:
                    end_words.append(words[-1].lower().rstrip('.,!?;:'))
        
        return {
            'end_char_distribution': Counter(end_chars).most_common(),
            'end_word_distribution': Counter(end_words).most_common(10),
            'texts_ending_with_period': sum(1 for char in end_chars if char == '.'),
            'texts_ending_with_question': sum(1 for char in end_chars if char == '?')
        }
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)