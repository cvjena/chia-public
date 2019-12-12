import json
import numpy as np
import matplotlib.pyplot as plt


def main(lines):
    result_storage = {}
    for filename, _, _ in lines:
        try:
            with open(filename) as target:
                data = json.load(target)
            run_count = len(data)
            print(f"{run_count} runs found.")

            print(f"{len(data[0])} reports per run.")

            # for run in data:
            #     run[-1] = [run[-1]]
            #     print(run[-1])
            # with open(filename, "w") as target:
            #     json.dump(data, target)
            # continue

            intermediate_report_count = len(data[0]) - 1
            metrics = list(data[0][0][0].keys())

            result_storage_file = {
                metric: np.zeros(shape=(run_count, intermediate_report_count))
                for metric in metrics
            }

            for run_id in range(run_count):
                for intermediate_report_id in range(intermediate_report_count):
                    for metric in metrics:
                        if hasattr(data[run_id][intermediate_report_id], "keys"):
                            value = data[run_id][intermediate_report_id][metric]
                        else:
                            value = data[run_id][intermediate_report_id][0][metric]
                        result_storage_file[metric][
                            run_id, intermediate_report_id
                        ] = value

            result_storage[filename] = result_storage_file

        except Exception as ex:

            print(f"Skipping file {filename}: {str(ex)}")
            continue

    plt.figure()
    for filename, metric_filter, display_name in lines:
        result_storage_file = result_storage[filename]
        metric_storage = result_storage_file[metric_filter].transpose()

        y = np.mean(metric_storage, axis=1)
        ystd = np.std(metric_storage, axis=1)
        plt.plot(range(metric_storage.shape[0]), y, label=display_name)
        plt.fill_between(
            range(metric_storage.shape[0]), y - 0.5 * ystd, y + 0.5 * ystd, alpha=0.5
        )

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    mode_arg = sys.argv[1]
    if mode_arg == "intermediate":
        relevant_args = sys.argv[2:]

        assert len(relevant_args) % 3 == 0
        lines = []
        while len(relevant_args) > 0:
            lines += [(relevant_args[0], relevant_args[1], relevant_args[2])]
            relevant_args = relevant_args[3:]
        main(lines)

    else:
        print(f"Unknown mode: {mode_arg}")
