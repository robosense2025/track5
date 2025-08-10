# Evaluation Metrics

The final rank is calculated by the weighted score of both phase 1 and phase 2. Phase1 is 0.5 and phase 2 is 0.5. We use the averaged score across different metrics in each phase to calculate the final rank.

## Phase 1 Metrics

We use 3D *AP@0.50 (R40)* for the **Car** class evaluated on **Drone** platform data.

## Phase 2 Metrics

We use 3D *AP@0.50 (R40)* for the **Car** class and 3D *AP@0.50 (R40)* for the **Pedestrian** class evaluated on **Quadruped** platform data.

The final score is calculated as follows:

$$
\text{Phase2\_Score} = \frac{Car(AP_{3D}^{0.5}@40) + Pedestrian(AP_{3D}^{0.5}@40)}{2}
$$

The final score is calculated as follows:

$$
\text{Final\_Score} = 0.5 \times \text{Phase1\_Score} + 0.5 \times \text{Phase2\_Score}
$$

# Submission Requirements

To ensure competition fairness and reproducibility, all participants must open-source their codebase and provide comprehensive instructions for reproducing results. Participants are encouraged to cross-verify results from other teams. All times mentioned are in **Anywhere on Earth (AoE)** timezone (UTC-12).

## Timeline

### Phase 1 Results Finalization
**Deadline: August 22nd, 2025**

- Select and show the final and reproducible Phase 1 results to the CodaBench leaderboard
- These results will be used for final ranking and help all participants to better plan for phase 2 based on the phase 1 results.
- **Results will be considered VALID if:**
  - The results are reproducible.
  - The results are added to the CodaBench leaderboard. Better results that are intentionally **hidden from the leaderboard** will **NOT** be considered for the final ranking after this deadline.
- **Results will be considered INVALID if:**
  - Leaderboard results are not reproducible.
  - Better results that are intentionally hidden.

### Private Codebase Preparation
**Deadline: September 5th, 2025**

- Prepare a **private** GitHub repository containing code to reproduce Phase 1 results
- Grant access to the RoboSense official GitHub account: `robosense2025`
- Our team will begin Phase 1 result reproduction at this time
- This step helps participants prepare for the final public codebase release

### Public Codebase Release
**Deadline: September 15th, 2025**

- Release the final public codebase containing:
  - Code to reproduce Phase 1 results
  - Code to reproduce Phase 2 results
- This ensures competition transparency and fairness
- The organizing team will attempt to reproduce all results
- Submission with rule violations will not be considered for the final ranking
- Participants are encouraged to actively review other teams' codebases and report rule violations
    - **Note:** Reports submitted after the official award announcement will not be considered