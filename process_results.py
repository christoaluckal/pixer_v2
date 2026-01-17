import os
kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kResultsFolder = kRootFolder + "/results"
import numpy as np

results = os.listdir(kResultsFolder)
results = [os.path.join(kResultsFolder, f) for f in results if not f.startswith(".placeholder")]

def process_evo_file(evo_lines):
    mean_= -1
    for i in range(len(evo_lines)):
        if "mean" in evo_lines[i]:
            mean_= i
            break
    if mean_ == -1:
        print("Mean APE not found in APE report.")
        return
    l = evo_lines[mean_].strip()
    ape_val = l.split("\t")[-1]
    return ape_val
    
def extract_stat_value(lines, key, round=4):
    temp_lines = [line for line in lines if key in line][0]
    temp_lines = temp_lines.strip()
    value = temp_lines.split(":")[-1].strip()
    try:
        value = np.around(float(value), round)
        return str(value)
    except:
        value = np.around(float(value.split(" ")[0]), round)
        return str(value)


def process_results():
    row = ["Experiment", "Feature_Num", "Feature_Type", "APE", "RPE", "Original_KPs", "Masked_KPs", "KP_Reduction", "Est_Times", "Total_Loop_Time"]
    with open("summary_results.csv", "w") as summary_file:
        summary_file.write(",".join(row) + "\n")
        for result in results:

            exp_type = result.split("/")[-1].split("_")[0]
            feature_num = result.split("$")[1]
            feature_type = result.split("#")[1]

            exp_name = f"{exp_type}_${feature_num}$_#{feature_type}#"
            evo_report_location = os.path.join(result, f"{exp_name}_evo_report")
            try:
                ape_report = os.path.join(evo_report_location, "ape.txt")
                ape_val = None
                with open(ape_report, "r") as f:
                    evo_lines = f.readlines()
                    ape_val = process_evo_file(evo_lines)
                
                rpe_val = None
                rpe_report = os.path.join(evo_report_location, "rpe.txt")
                with open(rpe_report, "r") as f:
                    evo_lines = f.readlines()
                    rpe_val = process_evo_file(evo_lines)

                stats_file = os.path.join(result, f"{exp_name}_stats.txt")
                stats_lines = None
                with open(stats_file, "r") as f:
                    stats_lines = f.readlines()

                original_kps_val = extract_stat_value(stats_lines, "original_kps", round=4)
                masked_kps_val = extract_stat_value(stats_lines, "masked_kps", round=4)

                kps_reduction = (1 - (float(masked_kps_val) / float(original_kps_val))) * 100
                kps_reduction = str(np.around(kps_reduction, 2))+"%"
                est_times_val = extract_stat_value(stats_lines, "est_times", round=4)
                total_loop_time_val = extract_stat_value(stats_lines, "total_loop_time", round=2)

                data_row = [exp_type, feature_num, feature_type, ape_val, rpe_val, original_kps_val, masked_kps_val, kps_reduction, est_times_val, total_loop_time_val]
                summary_file.write(",".join(data_row) + "\n")
            except Exception as e:
                error_row = [exp_type, feature_num, feature_type, "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR"]
                summary_file.write(",".join(error_row) + "\n")
                print(f"Error processing result {result}: {e}")

if __name__ == "__main__":
    process_results()