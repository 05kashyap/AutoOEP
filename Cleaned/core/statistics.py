"""
Statistics utilities for analyzing prediction results
"""


class StatisticsCalculator:
    """Handles calculation of statistics from prediction results"""
    
    def __init__(self):
        """Initialize statistics calculator"""
        self.frame_stats = []
        self.session_stats = {}
    
    def update_frame_stats(self, frame_results):
        """Update statistics with new frame results"""
        self.frame_stats.append(frame_results)
    
    def get_comprehensive_stats(self):
        """Get comprehensive statistics from all processed frames"""
        if not self.frame_stats:
            return {}
        
        return self.calculate_overall_statistics(self.frame_stats)
    
    def reset(self):
        """Reset all statistics"""
        self.frame_stats = []
        self.session_stats = {}
    
    @staticmethod
    def calculate_overall_statistics(results):
        """Calculate overall statistics from processing results"""
        temporal_predictions = [r['temporal_prediction'] for r in results if r['temporal_prediction'] is not None]
        xgboost_predictions = [r['xgboost_prediction'] for r in results if r['xgboost_prediction'] is not None]
        static_predictions = [r['static_model_prediction'] for r in results if r['static_model_prediction'] is not None]
        
        stats = {
            'total_frames': len(results),
            'temporal_stats': StatisticsCalculator._calculate_prediction_stats(temporal_predictions, "Temporal"),
            'xgboost_stats': StatisticsCalculator._calculate_prediction_stats(xgboost_predictions, "XGBoost"),
            'static_stats': StatisticsCalculator._calculate_prediction_stats(static_predictions, "Static Model")
        }
        
        return stats
    
    @staticmethod
    def _calculate_prediction_stats(predictions, model_name):
        """Calculate statistics for a specific model's predictions"""
        if not predictions:
            return None
        
        avg_prediction = sum(predictions) / len(predictions)
        max_prediction = max(predictions)
        min_prediction = min(predictions)
        above_threshold = sum(p > 0.5 for p in predictions) / len(predictions) * 100
        
        return {
            'model_name': model_name,
            'count': len(predictions),
            'average': avg_prediction,
            'maximum': max_prediction,
            'minimum': min_prediction,
            'above_threshold_percent': above_threshold
        }
    
    @staticmethod
    def print_statistics(stats):
        """Print formatted statistics"""
        print(f"\nProcessed {stats['total_frames']} frames")
        
        for stat_key in ['temporal_stats', 'xgboost_stats', 'static_stats']:
            stat = stats[stat_key]
            if stat is not None:
                print(f"\n{stat['model_name']} Statistics:")
                print(f"  Frames processed: {stat['count']}")
                print(f"  Average probability: {stat['average']:.4f}")
                print(f"  Maximum probability: {stat['maximum']:.4f}")
                print(f"  Minimum probability: {stat['minimum']:.4f}")
                print(f"  Above threshold (0.5): {stat['above_threshold_percent']:.2f}%")
