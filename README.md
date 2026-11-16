# Cost-Aware ATM Cash Replenishment using DQN

Reproducible implementation for NMAI511 Assignment B, continuing the Assignment A proposal.

## Dataset
Use **NN5 Daily Dataset (without Missing Values)**, Zenodo DOI `10.5281/zenodo.4656117` (111 UK ATM daily withdrawal series). Download and unzip the dataset, then point the program to the CSV.

## Run locally
```bash
pip install -r requirements.txt
python atm_dqn_assignment.py --data /path/to/nn5_daily_dataset_without_missing_values.csv --episodes 350
```

## Google Colab
Upload the script and NN5 CSV, then run:
```python
!pip -q install numpy pandas torch
!python atm_dqn_assignment.py --data nn5_daily_dataset_without_missing_values.csv --episodes 350
```

Outputs: `atm_dqn.pt`, `training_log.csv`, `test_results.csv`, and `test_summary.csv`.

## Design
State: normalized cash, previous 7 withdrawals, cyclical day-of-week, days since replenishment. Actions: 0 no replenish, 1 replenish fixed amount. Reward: negative replenishment + holding + unmet-demand cost. Training uses a DQN, replay buffer, target network, Huber loss, Adam, gradient clipping and decaying epsilon-greedy exploration. Final testing is greedy (epsilon=0). Chronological 70/15/15 split prevents future-data leakage. Baselines are fixed schedule and cash threshold.

## Reproducibility
Seed 42 is used. Report mean and standard deviation across evaluation windows; for a stronger submission, rerun training with multiple seeds and report cross-seed variability too.
