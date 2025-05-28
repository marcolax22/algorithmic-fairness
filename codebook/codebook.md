# Hiring Data Set

This codebook describes syntethic data on the hiring process of STEM education in the Netherlands. This dataset is designed to help educators and researchers explore and compare different definitions of algorithmic fairness. It offers a convenient way to illustrate concepts such as fairness through unawareness, group fairness, statistical parity, predictive parity, equalized odds, and treatment equality.

It includes multiple sensitive attributes—such as age, gender that can be used to define a variety of protected or sensitive groups.

## Source Information
van Otterloo, S. (2022). Utrecht fairness recruitment dataset. https://ictinstitute.nl/team/sieuwert-van-otterloo/

## Dataset Columns Description

This section provides detailed information on each column in the dataset used for analyzing algorithmic fairness in hiring decisions. The data consists of various demographic, educational, and experiential attributes of applicants, alongside the final hiring decision.

---

### `age`
- **Description**: The age of the applicant at the time of application.
- **Type**: Integer
- **Example**: `25`, `32`, `45`

---

### `gender`
- **Description**: The self-identified gender of the applicant, encoded numerically.
  - `0` = Male  
  - `1` = Female  
  - `2` = Other / Non-binary
- **Type**: Integer (Categorical Encoding)

---

### `ind-debateclub`
- **Description**: Indicates whether the applicant participated in a debate club.
- **Type**: Boolean
- **Values**: `True` or `False`

---

### `ind-entrepeneur_exp`
- **Description**: Indicates whether the applicant has entrepreneurial experience (e.g., started a business, freelanced, etc.).
- **Type**: Boolean
- **Values**: `True` or `False`

---

### `ind-exact_study`
- **Description**: Indicates whether the applicant pursued an academic field related to STEM (Science, Technology, Engineering, Mathematics).
- **Type**: Boolean
- **Values**: `True` or `False`

---

### `ind-international_exp`
- **Description**: Indicates whether the applicant has international experience, such as studying or working abroad.
- **Type**: Boolean
- **Values**: `True` or `False`

---

### `ind-languages`
- **Description**: Number of languages the applicant can speak fluently.
- **Type**: Integer
- **Typical Range**: `1` to `5`

---

### `ind-previous_exp`
- **Description**: Total years of prior work experience the applicant has.
- **Type**: Integer
- **Value Range**: `0` to `5`

---

### `ind-programming_exp`
- **Description**: Indicates whether the applicant has prior experience with programming or coding.
- **Type**: Boolean
- **Values**: `True` or `False`
---

### `ind-testresults`
- **Description**: Applicant’s score on an internal test administered as part of the recruitment process.
- **Type**: Integer
- **Score Range**: `0` to `50`

---

### `ind-university_grade`
- **Description**: Grade Point Average (GPA) or equivalent final university grade of the applicant, on a 0–100 scale.
- **Type**: Integer
- **Score Range**: `0` to `100`
- **Example**: `87`

---

### `nationality`
- **Description**: Nationality or citizenship of the applicant.
- **Type**: Categorical (Text)
- **Examples**: `Dutch`, `German`, `French`, `Spanish`, `Italian`

---

### `sport`
- **Description**: Main sport the applicant engages in or lists in their application.
- **Type**: Categorical (Text)
- **Examples**: `Football`, `Basketball`, `Tennis`, `Swimming`, `None`

---

### `decision`
- **Description**: Final hiring decision for the applicant.
- **Type**: Boolean
- **Values**: 
  - `True` = Applicant was hired  
  - `False` = Applicant was not hired
