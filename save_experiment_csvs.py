from experiment_common_code import save_to_csv
save_to_csv("test", "f1", "data/experiment_results/experiment_1", probe_type="lr")
save_to_csv("test", "f1", "data/experiment_results/experiment_1", probe_type="mm")
save_to_csv("test", "marginal_f1", "data/experiment_results/experiment_1", probe_type="lr")
save_to_csv("test", "marginal_f1", "data/experiment_results/experiment_1", probe_type="mm")

save_to_csv("test_a", "f1", "data/experiment_results/experiment_2", probe_type="lr")
save_to_csv("test_a", "f1", "data/experiment_results/experiment_2", probe_type="mm")
save_to_csv("test_b", "f1", "data/experiment_results/experiment_2", probe_type="lr")
save_to_csv("test_b", "f1", "data/experiment_results/experiment_2", probe_type="mm")