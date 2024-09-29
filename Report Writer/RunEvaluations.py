from helpers.Evaluator import evaluate_report
from helpers.OnePassWriter import one_pass_writer
from helpers.MoAWriter import MoAWriter
from tqdm import tqdm
import time
import pickle

class EvaluationItem:
    """
    This class represents an item to be evaluated in the evaluation framework.
    It encapsulates the model, data, template, and temperature for a single evaluation.
    """

    def __init__(self, model, data, template, n_trials=10, debug=True):
        """
        Initialize an EvaluationItem.

        Args:
            model: Information about model. ex: ollama/gemma:2b @ MoA 2"
            data: The data to be used in the evaluation.
            template: The template to be used for report generation.
            n_trials (int): Number of trials to run.
            debug (bool): Whether to print debug information.
        """
        self.trial_name = model
        self.data = data
        self.template = template
        self.n_trials = n_trials
        self.debug = debug
        # Method of Inference (1Pass, MoA, etc.)
        self.model = model.split("@")[0].strip()
        self.method = model.split("@")[1].strip().split(" ")[0]
        if self.method == "MoA":
            self.num_agents = int(model.split("@")[1].strip().split(" ")[1])
            self.MoAWriter = MoAWriter(self.model, self.data, self.template, self.num_agents)

    def run(self):
        """
        Run the evaluation for this item.

        This method generates multiple responses using the evaluate_report function,
        then aggregates these responses.

        Returns:
            The aggregated response from multiple evaluations.
        """
        acc = 0
        clarity = 0
        relevancy = 0
        style = 0
        responses = []
        avg_time = 0
        detailed_results = []

        for _ in tqdm(range(self.n_trials)):
            start_time = time.time()
            if self.method == "1Pass":
                response = one_pass_writer(self.model, self.data, self.template)
            elif self.method == "MoA":
                response = self.MoAWriter.run()
            end_time = time.time()

            if self.debug:
                print(response)

            evaluation = evaluate_report("gpt-4o-2024-08-06", response, self.data, self.template)
            cur_acc, cur_clarity, cur_relevancy, cur_style = map(float, evaluation.split(","))
            result = {
                "ratings": {
                    "accuracy": cur_acc,
                    "clarity": cur_clarity,
                    "relevancy": cur_relevancy,
                    "style": cur_style,
                    "avg": (cur_acc + cur_clarity + cur_relevancy + cur_style) / 4,
                },
                "time": end_time - start_time,
                "response": response
            }
            
            acc += result["ratings"]["accuracy"]
            clarity += result["ratings"]["clarity"]
            relevancy += result["ratings"]["relevancy"]
            style += result["ratings"]["style"]
            avg_time += result["time"]
            responses.append(result["response"])
            detailed_results.append(result)

        avg = (acc + clarity + relevancy + style) / (4 * self.n_trials)
        return {
            "avg_ratings" : {
                "accuracy": acc / self.n_trials,
                "clarity": clarity / self.n_trials,
                "relevancy": relevancy / self.n_trials,
                "style": style / self.n_trials,
                "avg": avg
            },
            "responses": responses,
            "time": avg_time / self.n_trials,
            "detailed": detailed_results
        }

class EvaluationFramework:
    """
    This class represents the framework for running evaluations on multiple items.
    """
    def __init__(self, models, data, template, n_trials=10, debug=False):
        """
        Initialize an EvaluationFramework.

        Args:
            models: A list of models to be evaluated.
            data: The data to be used in the evaluation.
            template: The template to be used for report generation.
            n_trials (int): Number of trials to run for each model.
            debug (bool): Whether to print debug information.
        """
        self.items = [EvaluationItem(model, data, template, n_trials, debug) for model in models]

    def run(self, debug=True):
        """
        Run the evaluation for all items.

        Returns:
            A list of results for each item.
        """
        results = {}

        for item in self.items:
            print(f"Running Trial {item.trial_name}")
            result = item.run()
            if debug:
                print(result["avg_ratings"])
            with open(f"./results/EVAL_{item.trial_name.split('/')[-1]}.pkl", "wb") as f:
                pickle.dump(result, f)
            results[item.model] = result

        return results

if __name__ == "__main__":
    import content
    models = [
        # "gpt-4o-mini @ 1Pass",
        # "gpt-4o-mini @ MoA 1",
        # "gpt-4o-mini @ MoA 2",
        # "gpt-4o-mini @ MoA 3",
        # "gpt-4o-mini @ MoA 4",
        # "gpt-4o-mini @ MoA 5",
        # "gpt-4o-mini @ MoA 6",
        # "gpt-4o-mini @ MoA 7",
        # "gpt-4o-mini @ MoA 8",
        # "gpt-4o-2024-08-06 @ 1Pass",
        # "gpt-4o-2024-08-06 @ MoA 1",
        # "gpt-4o-2024-08-06 @ MoA 2",
        # "gpt-4o-2024-08-06 @ MoA 3",
        # "gpt-4o-2024-08-06 @ MoA 4",
        # "gpt-4o-2024-08-06 @ MoA 5",
        # "gpt-4o-2024-08-06 @ MoA 6",
        # "gpt-4o-2024-08-06 @ MoA 7",
        # "gpt-4o-2024-08-06 @ MoA 8",
        # "claude-3-haiku @ 1Pass",
        # "claude-3-sonnet @ 1Pass",
        # "claude-3-opus @ 1Pass",
        # "claude-3-5-sonnet @ 1Pass",
        # "claude-3-5-sonnet @ MoA 1",
        # "claude-3-5-sonnet @ MoA 2",
        # "claude-3-5-sonnet @ MoA 3",
        # "claude-3-5-sonnet @ MoA 4",
        # "claude-3-5-sonnet @ MoA 5",
        # "claude-3-5-sonnet @ MoA 6",
        # "claude-3-5-sonnet @ MoA 7",
        # "claude-3-5-sonnet @ MoA 8",
        # "o1-mini @ 1Pass", # TO RUN
        # "o1-preview @ 1Pass", # TO RUN
        # "ollama/gemma2:2b @ 1Pass",
        # "ollama/gemma2:2b @ MoA 1",
        # "ollama/gemma2:2b @ MoA 2",
        # "ollama/gemma2:2b @ MoA 3",
        # "ollama/gemma2:2b @ MoA 4",
        # "ollama/gemma2:2b @ MoA 5",
        # "ollama/gemma2:2b @ MoA 6",
        # "ollama/gemma2:2b @ MoA 7",
        # "ollama/gemma2:2b @ MoA 8",
        # "ollama/qwen2:1.5b @ 1Pass",
        # "ollama/qwen2:1.5b @ MoA 1",
        # "ollama/qwen2:1.5b @ MoA 2",
        # "ollama/qwen2:1.5b @ MoA 3",
        # "ollama/qwen2:1.5b @ MoA 4",
        # "ollama/qwen2:1.5b @ MoA 5",
        # "ollama/qwen2:1.5b @ MoA 6",
        # "ollama/qwen2:1.5b @ MoA 7",
        # "ollama/qwen2:1.5b @ MoA 8",
        # "ollama/phi3.5:latest @ 1Pass",
        # "ollama/phi3.5:latest @ MoA 1",
        # "ollama/phi3.5:latest @ MoA 2",
        # "ollama/phi3.5:latest @ MoA 3",
        # "ollama/phi3.5:latest @ MoA 4",
        # "ollama/phi3.5:latest @ MoA 5",
        # "ollama/phi3.5:latest @ MoA 6",
        # "ollama/phi3.5:latest @ MoA 7",
        # "ollama/phi3.5:latest @ MoA 8",
        # "gemini-1.5-flash @ 1Pass",
        # "gemini-1.5-pro @ 1Pass",
        # "ollama/gemma2:2b @ 1Pass",
        # "ollama/gemma2:2b @ MoA 1",
        # "ollama/gemma2:2b @ MoA 2",
        # "ollama/gemma2:2b @ MoA 3",
        # "ollama/gemma2:2b @ MoA 4",
        # "ollama/gemma2:2b @ MoA 5",
        # "ollama/gemma2:2b @ MoA 6",
        # "ollama/gemma2:2b @ MoA 7",
        # "ollama/gemma2:2b @ MoA 8",
        # "ollama/qwen2.5:0.5b-instruct @ 1Pass",
        # "ollama/qwen2.5:0.5b-instruct @ MoA 1",
        # "ollama/qwen2.5:0.5b-instruct @ MoA 2",
        # "ollama/qwen2.5:0.5b-instruct @ MoA 3",
        # "ollama/qwen2.5:0.5b-instruct @ MoA 4",
        # "ollama/qwen2.5:0.5b-instruct @ MoA 5",
        # "ollama/qwen2.5:0.5b-instruct @ MoA 6",
        # "ollama/qwen2.5:0.5b-instruct @ MoA 7",
        # "ollama/qwen2.5:0.5b-instruct @ MoA 8",
        # "ollama/qwen2.5:1.5b-instruct @ 1Pass",
        # "ollama/qwen2.5:1.5b-instruct @ MoA 1",
        # "ollama/qwen2.5:1.5b-instruct @ MoA 2",
        # "ollama/qwen2.5:1.5b-instruct @ MoA 3",
        # "ollama/qwen2.5:1.5b-instruct @ MoA 4",
        # "ollama/qwen2.5:1.5b-instruct @ MoA 5",
        # "ollama/qwen2.5:1.5b-instruct @ MoA 6",
        # "ollama/qwen2.5:1.5b-instruct @ MoA 7",
        # "ollama/qwen2.5:1.5b-instruct @ MoA 8",
        # "ollama/qwen2.5:3b-instruct @ 1Pass",
        # "ollama/qwen2.5:3b-instruct @ MoA 1",
        # "ollama/qwen2.5:3b-instruct @ MoA 2",
        # "ollama/qwen2.5:3b-instruct @ MoA 3",
        # "ollama/qwen2.5:3b-instruct @ MoA 4",
        # "ollama/qwen2.5:3b-instruct @ MoA 5",
        # "ollama/qwen2.5:3b-instruct @ MoA 6",
        # "ollama/qwen2.5:3b-instruct @ MoA 7",
        # "ollama/qwen2.5:3b-instruct @ MoA 8",
    ]
    evaluation_framework = EvaluationFramework(
        models=models, 
        data=content.data_, 
        template=content.template_, 
        n_trials=10
    )
    results = evaluation_framework.run()
    import pickle
    with open("./results/ALL_RESULTS.pkl", "wb") as f:
        pickle.dump(results, f)