import os
import json
import pandas as pd
from datetime import datetime

class ResultsLogger:
    def __init__(self, log_dir="results"):
        """
        Initialize the results logger.
        
        Args:
            log_dir: Directory to store logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            "timestamp": self.timestamp,
            "model_name": None,
            "dataset_name": None,
            "aggregated_metrics": {},
            "question_results": []
        }
        
    def log_model_info(self, model_name, dataset_name):
        """Log model and dataset information"""
        self.results["model_name"] = model_name
        self.results["dataset_name"] = dataset_name
        
    def log_question_result(self, question, expected_answer, model_answer, question_type=None, correct=None):
        """
        Log individual question result
        
        Args:
            question: The question text
            expected_answer: The expected answer
            model_answer: The model's answer
            question_type: Type/category of the question
            correct: Whether the answer was correct
        """
        if correct is None:
            correct = (str(expected_answer).strip() == str(model_answer).strip())
            
        self.results["question_results"].append({
            "question": question,
            "expected_answer": expected_answer,
            "model_answer": model_answer,
            "question_type": question_type,
            "correct": correct
        })
        
    def log_aggregated_metrics(self, metrics):
        """Log overall evaluation metrics"""
        self.results["aggregated_metrics"] = metrics
        
    def compute_type_statistics(self):
        """Compute statistics by question type"""
        type_stats = {}
        
        for result in self.results["question_results"]:
            q_type = result.get("question_type", "unknown")
            
            if q_type not in type_stats:
                type_stats[q_type] = {"total": 0, "correct": 0, "incorrect": 0}
                
            type_stats[q_type]["total"] += 1
            if result["correct"]:
                type_stats[q_type]["correct"] += 1
            else:
                type_stats[q_type]["incorrect"] += 1
        
        # Calculate accuracy for each type
        for q_type in type_stats:
            total = type_stats[q_type]["total"]
            if total > 0:
                type_stats[q_type]["accuracy"] = type_stats[q_type]["correct"] / total
            else:
                type_stats[q_type]["accuracy"] = 0
                
        self.results["type_statistics"] = type_stats
        return type_stats

    def save_results(self, format="all"):
        """
        Save results to files
        
        Args:
            format: Output format ('json', 'csv', or 'all')
        """
        # Compute statistics if not already done
        if "type_statistics" not in self.results:
            self.compute_type_statistics()
            
        # Create filenames with proper directory structure
        model_dir = self.results['model_name'].replace('/', '_')  # Replace slashes in model name
        results_dir = os.path.join(self.log_dir, model_dir)
        
        # Ensure directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        base_filename = f"{self.results['model_name'].replace('/', '_')}_{self.results['dataset_name']}_{self.timestamp}"
        json_path = os.path.join(results_dir, f"{base_filename}.json")
        csv_path = os.path.join(results_dir, f"{base_filename}.csv")
        stats_path = os.path.join(results_dir, f"{base_filename}_stats.csv")
        
        if format in ["json", "all"]:
            with open(json_path, "w") as f:
                json.dump(self.results, f, indent=2)
                
        if format in ["csv", "all"]:
            # Save question results to CSV
            df = pd.DataFrame(self.results["question_results"])
            df.to_csv(csv_path, index=False)
            
            # Save type statistics to CSV
            stats_data = []
            for q_type, stats in self.results["type_statistics"].items():
                stats_data.append({
                    "question_type": q_type,
                    **stats
                })
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_csv(stats_path, index=False)
            
        return {
            "json_path": json_path if format in ["json", "all"] else None,
            "csv_path": csv_path if format in ["csv", "all"] else None,
            "stats_path": stats_path if format in ["csv", "all"] else None
        }