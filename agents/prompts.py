SUMMARIZE_OBSERVATION_TEMPLATE = """{system_tag_start}You are a medical artificial intelligence assistant. Your goal is to effectively, efficiently and accurately reduce text without inventing information. You want to return verbatim observations that are abnormal and of interest to a possible diagnosis of the patient. Normal observations can be combined. Do not invent information. Use medical abbreviations when possible to save characters. Put the most important information first.{system_tag_end}{user_tag_start}Please summarize the following result:
{observation}{user_tag_end}{ai_tag_start}
Summary: """

FULL_INFO_TEMPLATE_COT = """{system_tag_start}You are a medical artificial intelligence assistant. You diagnose patients based on the provided information to assist a doctor in his clinical duties. Your goal is to correctly diagnose the patient. Based on the provided information you will provide a final diagnosis of the most severe pathology. Consider the facts of the case first, step by step.{system_tag_end}{fewshot_examples}{user_tag_start}Consider the following case:

{diagnostic_guidelines}{diagnostic_criteria}{input}\n\nWhat is the final diagnosis?{user_tag_end}{ai_tag_start}\nLets think step by step"""

FULL_INFO_TEMPLATE_COT_FINAL_DIAGNOSIS = """{system_tag_start}You are a medical artificial intelligence assistant. You diagnose patients based on the provided information to assist a doctor in his clinical duties. Your goal is to correctly diagnose the patient. Based on the provided information you will provide a final diagnosis of the most severe pathology. Consider the facts of the case first, and then provide the diagnosis. Give only a single diagnosis.{system_tag_end}{user_tag_start}Consider the following case summary and then provide the most likely final diagnosis.

{cot}{user_tag_end}{ai_tag_start}\nFinal Diagnosis:"""

FULL_INFO_TEMPLATE = """{system_tag_start}You are a medical artificial intelligence assistant. You directly diagnose patients based on the provided information to assist a doctor in his clinical duties. Your goal is to correctly diagnose the patient. Based on the provided information you will provide a final diagnosis of the most severe pathology. Don't write any further information. Give only a single diagnosis.{system_tag_end}{fewshot_examples}{user_tag_start}Provide the most likely final diagnosis of the following patient.

{input}{user_tag_end}{ai_tag_start}Final Diagnosis:"""

FULL_INFO_TEMPLATE_DIAGSUM = """{system_tag_start}You are a medical artificial intelligence assistant. You directly diagnose patients based on the provided information to assist a doctor in his clinical duties. Your goal is to correctly diagnose the patient.  Based on the provided information you will provide the diagnostic criteria of the most confident pathology. Don't write any further information. Provide a single diagnosis along with the most relevant diagnostic criteria.{system_tag_end}{fewshot_examples}{user_tag_start}Provide the most relevant diagnostic criteria of the following patient with no more other information.

{diagnostic_guidelines}{diagnostic_criteria}{input}{user_tag_end}{ai_tag_start}Final Diagnosis and diagnostic criteria:"""

FULL_INFO_TEMPLATE_DIAGSUM_WITH_PAST = """{system_tag_start}You are a senior medical artificial intelligence assistant. Your primary duty is to assist a doctor by verifying previous diagnostic opinions and forming an independent diagnosis based on the patient's complete situation. Based on the provided information you will provide the diagnostic criteria of the most confident pathology. Don't write any further information. Provide a single diagnosis along with the most relevant diagnostic criteria.{system_tag_end}{fewshot_examples}{user_tag_start}Provide the most relevant diagnostic criteria of the following patient with no more other information.

{diagnostic_guidelines}{input}

The provided diagnostic results are for reference only. Given that prior opinions may be inconsistent, you must conduct an impartial evaluation and not simply echo others' findings. 
@@@ Past Diagnosis Results @@@
{past_diagnosis_results}

{user_tag_end}{ai_tag_start}Final Diagnosis and diagnostic criteria:"""

FULL_INFO_TEMPLATE_HUMAN = """
Consider the following case. What is the most likely diagnosis? Give a single diagnosis. Do not write any further information.

{input}
Final Diagnosis: """


FI_FEWSHOT_TEMPLATE_COPD_RR = """{user_tag_start}Provide the most likely final diagnosis of the following patient.

@@@ PATIENT HISTORY @@@
Mrs. ___ is a 69-year old women who called an ambulance due to acute shortness of breath. The patient suddenly became dyspneic and is in fear of suffocating. The patient denies chest pain, abdominal pain, vomiting or fever. Past medical history: HTN, COPD, Smoker. Social History: __ Family History: Uncle and father died of lung disease.

@@@ PHYSICAL EXAMINATION @@@
Observation: T 98.3,BP 145/83, HR 110, RR 25, SPO2 85% RA Wgt 73 kg Gen: Dyspnoe. A+Ox3; CV: RRR, no MGR; Lungs: Difficult exam due to patient noncompliance with breaths, general wheezing, dry coughs during examination.

@@@ LABORATORY RESULTS @@@
(<FLUID>) <TEST>: <RESULT> | REFERENCE RANGE (RR): [LOWER RR - UPPER RR]
(Blood) WBC: 13.7 K/uL | RR: [4.0 - 10.0]
(Blood) RBC: 3.51 m/uL | RR: [3.9 - 5.2]
(Blood) Hgb: 11.4 g/dL | RR: [11.2 - 15.7]
(Blood) Hct: 29.4 % | RR: [34.0 - 45.0]
(Blood) pCO2: 67.4 mm Hg | RR: [35.0 - 45.0]
(Blood) pO2: 61.5 mm Hg | RR: [85.0 - 105.0]

@@@ IMAGING RESULTS @@@
Observation: CXR: Relative lucency of the upper to mid lungs suggests underlying pulmonary emphysema. Bibasilar atelectasis. No overt pulmonary edema, no large pleural effusion or pneumothorax. Cardiac and mediastinal silhouettes are unremarkable.{user_tag_end}{ai_tag_start}Final Diagnosis: Acute exacerbation of COPD{ai_tag_end}"""

FI_FEWSHOT_TEMPLATE_PNEUMONIA_RR = """{user_tag_start}Provide the most likely final diagnosis of the following patient.

@@@ PATIENT HISTORY @@@
Mr. _ is a 63-year old male presenting with cough and fever. He states that the symptoms began four days ago with sore throat, rhinorrhea and fatigue. Since this morning he experiences shortness of breath while coughing, which is his reason for admission. He denies chest pain, headache, nausea, abdominal pain, diarrhea or urinary complaints. Past medical history: Gastroesophageal reflux disease, Hypertension Social History:____ Family History: Significant for diabetes and coronary artery disease.

@@@ PHYSICAL EXAMINATION @@@
Observation: Physical Examination upon admission: T 101.4,BP 152/78, HR 88, RR 23, SPO2 97% RA Wgt 98 kg; Gen: NAD. A+Ox3; HEENT: PERRL, EOMI, MMM, Oropharynx clear; Neck: Supple, no LAD, no JVP elevation; CV: RRR, no MGR; Lungs: Rales and rhonchi at right base, no wheeze, coughing during examination;

@@@ LABORATORY RESULTS @@@
(<FLUID>) <TEST>: <RESULT> | REFERENCE RANGE (RR): [LOWER RR - UPPER RR]
(Blood) WBC: 13 K/uL | RR: [4.0 - 10.0]
(Blood) RBC: 4.6 m/uL | RR: [3.9 - 5.2]
(Blood) MCV: 88 fL | RR: [82.0 - 98.0]
(Blood) MCH: 31.3 pg | RR: [27.0 - 32.0]
(Blood) HbA1c: 7.1 % | RR: [4.0 - 6.0]

@@@ IMAGING RESULTS @@@
CXR: Bilateral basal ground glass opacities and scattered consolidation in the RLL. No large pleural effusion or evidence of pneumothorax is seen. The cardiac and mediastinal silhouettes are unremarkable.{user_tag_end}{ai_tag_start}Final diagnosis: Pneumonia{ai_tag_end}"""

FI_FEWSHOT_TEMPLATE_COPD = """{user_tag_start}Provide the most likely final diagnosis of the following patient.

@@@ PATIENT HISTORY @@@
Mrs. ___ is a 69-year old women who called an ambulance due to acute shortness of breath. The patient suddenly became dyspneic and is in fear of suffocating. The patient denies chest pain, abdominal pain, vomiting or fever. Past medical history: HTN, COPD, Smoker. Social History: __ Family History: Uncle and father died of lung disease.

@@@ PHYSICAL EXAMINATION @@@
Observation: T 98.3,BP 145/83, HR 110, RR 25, SPO2 85% RA Wgt 73 kg Gen: Dyspnoe. A+Ox3; CV: RRR, no MGR; Lungs: Difficult exam due to patient noncompliance with breaths, general wheezing, dry coughs during examination.

@@@ LABORATORY RESULTS @@@
(<FLUID>) <TEST>: <RESULT>
(Blood) WBC: 13.7 K/uL
(Blood) RBC: 3.51 m/uL
(Blood) Hgb: 11.4 g/dL
(Blood) Hct: 29.4 %
(Blood) pCO2: 67.4 mm Hg
(Blood) pO2: 61.5 mm Hg

@@@ IMAGING RESULTS @@@
Observation: CXR: Relative lucency of the upper to mid lungs suggests underlying pulmonary emphysema. Bibasilar atelectasis. No overt pulmonary edema, no large pleural effusion or pneumothorax. Cardiac and mediastinal silhouettes are unremarkable.{user_tag_end}{ai_tag_start}Final Diagnosis: Acute exacerbation of COPD{ai_tag_end}"""

FI_FEWSHOT_TEMPLATE_PNEUMONIA = """{user_tag_start}Provide the most likely final diagnosis of the following patient.

@@@ PATIENT HISTORY @@@
Mr. _ is a 63-year old male presenting with cough and fever. He states that the symptoms began four days ago with sore throat, rhinorrhea and fatigue. Since this morning he experiences shortness of breath while coughing, which is his reason for admission. He denies chest pain, headache, nausea, abdominal pain, diarrhea or urinary complaints. Past medical history: Gastroesophageal reflux disease, Hypertension Social History:____ Family History: Significant for diabetes and coronary artery disease.

@@@ PHYSICAL EXAMINATION @@@
Observation: Physical Examination upon admission: T 101.4,BP 152/78, HR 88, RR 23, SPO2 97% RA Wgt 98 kg; Gen: NAD. A+Ox3; HEENT: PERRL, EOMI, MMM, Oropharynx clear; Neck: Supple, no LAD, no JVP elevation; CV: RRR, no MGR; Lungs: Rales and rhonchi at right base, no wheeze, coughing during examination;

@@@ LABORATORY RESULTS @@@
(<FLUID>) <TEST>: <RESULT>
(Blood) WBC: 13 K/uL
(Blood) RBC: 4.6 m/uL
(Blood) MCV: 88 fL
(Blood) MCH: 31.3 pg
(Blood) HbA1c: 7.1 %

@@@ IMAGING RESULTS @@@
CXR: Bilateral basal ground glass opacities and scattered consolidation in the RLL. No large pleural effusion or evidence of pneumothorax is seen. The cardiac and mediastinal silhouettes are unremarkable.{user_tag_end}{ai_tag_start}Final Diagnosis: Pneumonia{ai_tag_end}"""

FI_FEWSHOT_TEMPLATE_APPENDICITIS = """{user_tag_start}Provide the most likely final diagnosis of the following patient.

@@@ PATIENT HISTORY @@@
Ms. ___ is a 26-year-old woman presenting with right lower quadrant abdominal pain for the past 18 hours. The pain started as a vague discomfort around the umbilicus, later localizing to the right lower abdomen. She also reports nausea and a single episode of vomiting. No diarrhea or urinary symptoms. Last menstrual period was 12 days ago. Past medical history: Unremarkable. Social history: College student, non-smoker, no alcohol. Family history: Non-contributory.

@@@ PHYSICAL EXAMINATION @@@
Observation: T 100.9°F, BP 124/76, HR 94, RR 18, Wgt 60 kg; Gen: Appears mildly distressed due to abdominal pain; Abd: Tenderness in RLQ with guarding, rebound tenderness present. Positive Rovsing's sign and obturator sign. No palpable masses or hepatosplenomegaly.

@@@ LABORATORY RESULTS @@@
(<FLUID>) <TEST>: <RESULT>
(Blood) WBC: 14.2 K/uL
(Blood) Neutrophils: 89%
(Blood) CRP: 6.5 mg/dL
(Blood) Hgb: 12.1 g/dL
(Urine) hCG: Negative

@@@ IMAGING RESULTS @@@
CT Abdomen with contrast: Distended, fluid-filled appendix measuring 10 mm in diameter with surrounding fat stranding. No perforation or abscess noted. No free air in the abdomen.{user_tag_end}{ai_tag_start}Final Diagnosis: Acute Appendicitis{ai_tag_end}"""

FI_FEWSHOT_TEMPLATE_CHOLECYSTITIS = """{user_tag_start}Provide the most likely final diagnosis of the following patient.

@@@ PATIENT HISTORY @@@
Mr. ___ is a 52-year-old male presenting with right upper quadrant abdominal pain for the past 2 days. The pain is constant, dull, and worsens after eating greasy food. He reports associated nausea and low-grade fever. No vomiting, diarrhea, or urinary symptoms. Past medical history: Hyperlipidemia, obesity. Social history: Former smoker, occasional alcohol use. Family history: Mother had gallstones.

@@@ PHYSICAL EXAMINATION @@@
Observation: T 100.8°F, BP 138/84, HR 90, RR 20, Wgt 102 kg; Gen: Mild distress; Abd: Tenderness in the RUQ with positive Murphy's sign. No rebound tenderness or peritoneal signs. Bowel sounds present.

@@@ LABORATORY RESULTS @@@
(<FLUID>) <TEST>: <RESULT>
(Blood) WBC: 13.8 K/uL
(Blood) AST: 42 U/L
(Blood) ALT: 38 U/L
(Blood) ALP: 160 U/L
(Blood) Total Bilirubin: 1.8 mg/dL
(Blood) CRP: 5.2 mg/dL

@@@ IMAGING RESULTS @@@
Abdominal Ultrasound: Gallbladder wall thickening (5.5 mm), pericholecystic fluid, and presence of gallstones. Positive sonographic Murphy's sign. No bile duct dilation.{user_tag_end}{ai_tag_start}Final Diagnosis: Acute Cholecystitis{ai_tag_end}"""

FULL_INFO_TEMPLATE_DIAGSUM_WITH_DIFFERENCE = """{system_tag_start}You are a senior medical artificial intelligence assistant. Your primary duty is to assist a doctor by verifying previous diagnostic opinions and forming an independent diagnosis based on the patient's complete situation. Based on the provided information you will provide the diagnostic criteria of the most confident pathology. Don't write any further information. Provide a single diagnosis along with the most relevant diagnostic criteria.{system_tag_end}{fewshot_examples}{user_tag_start}Provide the most relevant diagnostic criteria of the following patient with no more other information.

{diagnostic_criteria}{input}

The provided diagnostic results are for reference only. Given that prior opinions may be inconsistent, you must conduct an impartial evaluation and not simply echo others' findings. 
@@@ PAST DIAGNOSIS RESULTS @@@
{past_diagnosis_results}

The following are significant clinical differences in related diseases for reference. Given that these contents may not fully match the actual situation, you need to conduct a fair evaluation instead of relying on these references for arbitrary diagnosis.
@@@ DIFFERENCES BETWEEN DISEASES @@@

{difference}

{user_tag_end}{ai_tag_start}Final Diagnosis and diagnostic criteria:"""