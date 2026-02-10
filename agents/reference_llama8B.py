cholecystitis = """
Cholecystitis general:
Gallbladder wall appearance is 'indistinct and hazy' suggesting chronic inflammation
No evidence of acute cholecystitis complications such as perforation, gangrene, or abscess formation
Right upper quadrant (RUQ) and epigastric pain
Presence of gallstones
Normal enhancement pattern of the liver and absence of intra-hepatic biliary obstruction suggest that this condition is not related to obstructive jaundice
Cholecystitis rare:
Circumferential wall edema up to 14mm
Questionable focal wall interruption at the gallbladder fundus
Adenomyomatosis (comet tail artifacts along the gallbladder wall)
Presence of multiple gallstones (>7)
Presence of a small subhepatic/pericholecystic fluid collection
Heterogeneous appearance and increased vascularity of the gallbladder wall"""


appendicitis = """
Appendicitis general:
Initial periumbilical abdominal pain that localizes to the Right Lower Quadrant (RLQ)
Some individuals experience general discomfort, fatigue, or lack of interest in food before experiencing more severe symptoms.
Change in Stool Consistency.
Nausea/Vomiting.
Imaging findings of a dilated appendix
Localized RLQ tenderness with rebound and involuntary guarding
Appendicitis rare:
Recurrent episodes of RLQ abdominal pain over an extended period
Cases might also feature secondary infections or co-morbidities affecting overall patient health
Blockage of the appendix can lead to its dilation and subsequent rupture
Pain migration patterns vary among individuals, sometimes beginning centrally before moving to the RLQ
Certain cases involve retrocecal positioning, increasing the likelihood of perforation"""



pancreatitis = """
Pancreatitis general:
Elevated serum lipase level (>3 times the upper limit of normal)
Severe epigastric pain radiating to back, often worse after eating.
Presence of gallstone in the distal Common Bile Duct (CBD)
Recent increased consumption of alcohol
Markedly increased serum amylase/lipase levels indicative of pancreatic damage/enzyme leakage into bloodstream.
Pancreatitis rare:
Elevated lipase level
Change in stool pattern from tan and solid to dark and loose
Presence of diffuse dilation of the common bile duct with numerous filling defects on ERCP imaging"""



diverticulitis = """
Diverticulitis general:
Imaging studies showing multiple diverticula in the distal thickened sigmoid colon
Increased white blood cell count indicative of systemic inflammatory response syndrome.
Presence of significant stranding around the sigmoid colon
Diverticulitis rare:
Thickening of the sigmoid colon wall (>50%)
Extraluminal air without organized collection, suggesting recent perforation
Partial or complete blockage of the intestines by swollen segments, causing severe vomiting, abdominal tenderness, and cessation of gas passage.
Loss of fat plane between the left ovary and the colon"""




pericarditis = """
Pericarditis general:
Unilateral chest pain worsening with deep breathing and exertion
Electrocardiographic Changes: Specific electrocardiogram patterns like low voltage QRS complexes (<5mm), diffuse ST elevation without reciprocal changes, or PR interval prolongation.
A distinctive friction rub heard over the precordium during auscultation.
Elevated inflammatory markers (CRP > 10 mg/L)
Mild tortuosity of the thoracic aorta
Enlarged cardiac silhouette
Pericarditis rare:
Presence of pulsus paradoxus (> 10 mmHg)
Normal echocardiogram showing no evidence of cardiac tamponade or ventricular dysfunction
Positive family history of autoimmune diseases
Mild metabolic acidosis and elevated white blood cell count suggesting inflammation
Resolution of symptoms after drainage of the pericardial fluid
Recent history of trauma (bike accident)"""


pneumonia ="""
Pneumonia general:
Respiratory Failure: Characterized by difficulty breathing, low oxygen saturation, and increased work of breathing, typically developing progressively.
Fever: Temperature exceeding 38°C was noted in most cases, often with associated chills.
Radiographic evidence of consolidation: Lobar or segmental consolidation visible on imaging.
Physical examination findings: Coarse breath sounds, absent or diminished breath sounds, and localized crackles were detected, especially over the site of consolidation.
Increased White Blood Cell Count: A hallmark sign of infection, often exceeding 12,000 cells/microL with neutrophilic predominance.
Altered Mental Status: Suggestive of systemic infection affecting brain function, particularly in older or severely ill patients.
Productive cough: Cough producing purulent sputum.
Pneumonia rare:
Presence of Infection Markers: Such as elevated CRP and procalcitonin, indicative of ongoing bacterial infection rather than thrombotic inflammation.
Patchy ill-defined opacity in the right lung base: Typically represents infection-related inflammation.
Mild leukopenia: Some patients, especially immunocompromised, showed a decrease in total white blood cell count.
Severe Hypoxemia: Extremely low oxygen saturation levels in the blood, necessitating urgent treatment.
Severe Bronchospasm: Can accompany pneumonia in patients with underlying reactive airway disease.
Presence of leukocytes in urine: Reflects systemic inflammatory response.
Patchy infiltration: Ground-glass patches and multifocal consolidation were visible on CT scans, suggestive of infectious."""


pulmonary_embolism = """
Pulmonary Embolism general:
Sudden onset of shortness of breath
Risk Factors: History of recent surgery, immobility, and previous trauma to the hips.
Absence of signs of right heart strain on echocardiogram.
Recent history of deep vein thrombosis (DVT).
Underlying Condition: Presence of mixed connective tissue disease.
Elevated D-Dimer Levels: Extremely high d-dimer levels exceeding expected ranges.
Markedly low blood pressure requiring vasopressor support.
Pulmonary Embolism rare:
Markedly low blood pressure requiring vasopressor support.
Low molecular weight heparin therapy discontinued before symptom onset.
Low Platelet Count: Platelet count significantly below normal range.
Simple renal cysts
Echocardiographic evidence of increased RV pressure leading to dilatation.
Small Right Pleural Effusion: Presence of trace right pleural effusion.
Reflux of contrast within the IVC and hepatic veins."""



chest_reference = "\n".join([pericarditis, pneumonia, pulmonary_embolism])
abdomen_reference = "\n".join(
    [cholecystitis, appendicitis, pancreatitis, diverticulitis]
)