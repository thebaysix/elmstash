"""
Report Generator - Creates comprehensive reports from observation and evaluation data.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


class ReportGenerator:
    """Generates comprehensive reports combining observation and evaluation data."""
    
    def __init__(self):
        self.report_templates = {
            'executive_summary': self._generate_executive_summary,
            'technical_analysis': self._generate_technical_analysis,
            'comparative_report': self._generate_comparative_report
        }
    
    def generate_comprehensive_report(
        self,
        observed_data: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        report_type: str = 'comprehensive'
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report combining observation and evaluation.
        
        Args:
            observed_data: Data from observer
            evaluation_results: Results from evaluator
            report_type: Type of report to generate
            
        Returns:
            Comprehensive report dictionary
        """
        return {
            'report_metadata': {
                'report_type': report_type,
                'generated_at': datetime.now().isoformat(),
                'generator_version': 'v1.0'
            },
            'executive_summary': self._generate_executive_summary(observed_data, evaluation_results),
            'technical_analysis': self._generate_technical_analysis(observed_data, evaluation_results),
            'recommendations': self._extract_recommendations(evaluation_results),
            'appendices': {
                'raw_observations': observed_data,
                'detailed_evaluations': evaluation_results
            }
        }
    
    def _generate_executive_summary(
        self, 
        observed_data: Dict[str, Any], 
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive summary section."""
        interactions = observed_data.get('interactions', [])
        overall_assessment = evaluation_results.get('overall_assessment', {})
        
        return {
            'key_metrics': {
                'total_interactions': len(interactions),
                'overall_score': overall_assessment.get('overall_score', 0),
                'quality_rating': overall_assessment.get('quality_rating', 'unknown')
            },
            'summary_statement': self._generate_summary_statement(overall_assessment),
            'critical_findings': self._extract_critical_findings(evaluation_results)
        }
    
    def _generate_technical_analysis(
        self, 
        observed_data: Dict[str, Any], 
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate technical analysis section."""
        metrics = observed_data.get('metrics', {})
        patterns = observed_data.get('patterns', {})
        capabilities = evaluation_results.get('capabilities', {})
        alignment = evaluation_results.get('alignment', {})
        
        return {
            'objective_measurements': {
                'entropy_analysis': metrics.get('response_entropy', 0),
                'diversity_metrics': {
                    'response_uniqueness': metrics.get('response_uniqueness_ratio', 0),
                    'unique_responses': metrics.get('unique_responses', 0)
                },
                'pattern_detection': self._summarize_patterns(patterns)
            },
            'performance_evaluation': {
                'capability_scores': self._extract_capability_scores(capabilities),
                'alignment_scores': self._extract_alignment_scores(alignment)
            },
            'statistical_insights': self._generate_statistical_insights(metrics, patterns)
        }
    
    def _generate_comparative_report(
        self, 
        comparative_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comparative analysis report."""
        return {
            'comparison_summary': {
                'models_compared': len(comparative_data),
                'comparison_timestamp': datetime.now().isoformat()
            },
            'relative_performance': self._analyze_relative_performance(comparative_data),
            'strengths_weaknesses': self._identify_comparative_strengths_weaknesses(comparative_data)
        }
    
    def _generate_summary_statement(self, overall_assessment: Dict[str, Any]) -> str:
        """Generate a summary statement based on overall assessment."""
        quality_rating = overall_assessment.get('quality_rating', 'unknown')
        overall_score = overall_assessment.get('overall_score', 0)
        
        if quality_rating == 'excellent':
            return f"Model demonstrates excellent performance with an overall score of {overall_score:.2f}."
        elif quality_rating == 'good':
            return f"Model shows good performance with an overall score of {overall_score:.2f}."
        elif quality_rating == 'fair':
            return f"Model exhibits fair performance with an overall score of {overall_score:.2f}."
        else:
            return f"Model performance is below expectations with an overall score of {overall_score:.2f}."
    
    def _extract_critical_findings(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Extract critical findings from evaluation results."""
        findings = []
        
        overall_assessment = evaluation_results.get('overall_assessment', {})
        if overall_assessment.get('overall_score', 0) < 0.5:
            findings.append("Overall performance is below acceptable thresholds")
        
        capabilities = evaluation_results.get('capabilities', {})
        if isinstance(capabilities, dict):
            for capability, result in capabilities.items():
                if isinstance(result, dict) and result.get('assessment') == 'poor':
                    findings.append(f"Poor performance in {capability.replace('_', ' ')}")
        
        return findings
    
    def _summarize_patterns(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize detected patterns."""
        repetition_patterns = patterns.get('repetition_patterns', {})
        consistency_patterns = patterns.get('consistency_patterns', {})
        
        return {
            'repetition_detected': not repetition_patterns.get('no_data', True),
            'uniqueness_ratio': repetition_patterns.get('uniqueness_ratio', 0),
            'consistency_score': consistency_patterns.get('consistency_score', 0)
        }
    
    def _extract_capability_scores(self, capabilities: Dict[str, Any]) -> Dict[str, float]:
        """Extract capability scores."""
        scores = {}
        
        for capability, result in capabilities.items():
            if isinstance(result, dict) and 'score' in result:
                scores[capability] = result['score']
        
        return scores
    
    def _extract_alignment_scores(self, alignment: Dict[str, Any]) -> Dict[str, float]:
        """Extract alignment scores."""
        scores = {}
        
        for alignment_aspect, result in alignment.items():
            if isinstance(result, dict) and 'score' in result:
                scores[alignment_aspect] = result['score']
        
        return scores
    
    def _generate_statistical_insights(
        self, 
        metrics: Dict[str, Any], 
        patterns: Dict[str, Any]
    ) -> List[str]:
        """Generate statistical insights."""
        insights = []
        
        # Entropy insights
        response_entropy = metrics.get('response_entropy', 0)
        if response_entropy < 1.0:
            insights.append("Low response entropy indicates limited diversity in outputs")
        elif response_entropy > 3.0:
            insights.append("High response entropy indicates good diversity in outputs")
        
        # Uniqueness insights
        uniqueness_ratio = metrics.get('response_uniqueness_ratio', 0)
        if uniqueness_ratio < 0.5:
            insights.append("Low uniqueness ratio suggests repetitive responses")
        elif uniqueness_ratio > 0.9:
            insights.append("High uniqueness ratio indicates diverse, non-repetitive responses")
        
        return insights
    
    def _extract_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Extract recommendations from evaluation results."""
        recommendations = []
        
        overall_assessment = evaluation_results.get('overall_assessment', {})
        if 'recommendation' in overall_assessment:
            recommendations.append(overall_assessment['recommendation'])
        
        # Extract from capabilities
        capabilities = evaluation_results.get('capabilities', {})
        if isinstance(capabilities, dict):
            for capability, result in capabilities.items():
                if isinstance(result, dict) and 'improvement_suggestions' in result:
                    recommendations.extend(result['improvement_suggestions'])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _analyze_relative_performance(self, comparative_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze relative performance across models."""
        # This is a simplified implementation
        return {
            'performance_ranking': list(comparative_data.keys()),
            'score_differences': 'Analysis would go here'
        }
    
    def _identify_comparative_strengths_weaknesses(
        self, 
        comparative_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Identify comparative strengths and weaknesses."""
        # This is a simplified implementation
        return {
            'relative_strengths': [],
            'relative_weaknesses': []
        }