cholecystitis = """
Cholecystitis general:
Positive sonographic Murphy's sign
Right upper quadrant (RUQ) pain, often severe and persistent
Elevated white blood cell count (leukocytosis)
Gallbladder wall thickening (>4mm) with pericholecystic fluid
Presence of gallstones on imaging
Mild elevation of liver enzymes (ALT/AST)
Cholecystitis rare:
Proteinuria without significant hematuria
Significantly elevated liver enzymes (ALT/AST) in the presence of choledocholithiasis
Pericholecystic fluid collection, sometimes indicating gallbladder perforation or evolving abscess
Layering sludge in the gallbladder without visible gallstones, seen in acalculous cholecystitis
Rim sign on CT 
Bile duct dilation with documented stones"""



appendicitis = """
Appendicitis general:
Right lower quadrant abdominal pain
Tenderness at McBurney's point
Elevated white blood cell count (ranging from 9.9 to 27.5 K/uL)
Appendiceal wall thickening and luminal distension observed on CT imaging
Nausea and vomiting
Periappendiceal fat stranding on imaging
Enlarged appendix (size varying from 6mm to over 1 cm)
Appendicitis rare:
Presence of appendicoliths on imaging
Free pelvic fluid without overt signs of perforation
Abscess formation or fluid collections
Reactive periappendiceal lymph nodes
Reactive terminal ileitis
Absence of free air ruling out perforation"""


pancreatitis = """
Pancreatitis general:
Elevated Amylase/Lipase >3 times the upper limit of normal
Severe epigastric abdominal pain radiating to the back, often worsened by eating
Persistent nausea and vomiting, often leading to dehydration
Leukocytosis (elevated white blood cell count)
Abdominal tenderness upon physical examination, but no rebound tenderness unless complicated by necrosis or infection
Peripancreatic fluid collections and fat stranding with necrosis on CT
Pancreatitis rare:
Severe nausea and vomiting leading to profound dehydration and electrolyte imbalances
Biliary dilation or sludge without evidence of gallstone obstruction, suggesting biliary pancreatitis
Peripancreatic fat necrosis confirmed by imaging, often with pancreatic parenchymal changes
Unintentional weight loss (e.g., 30 lbs over three weeks) due to prolonged illness or exocrine insufficiency
Leukoysis (High WBC Count)
History of gallstone-related symptoms without acute cholecystitis features, but with signs of biliary pancreatitis"""


diverticulitis = """
Diverticulitis general:
Left lower quadrant (LLQ) abdominal tenderness on physical examination
CT scan revealing diverticula and potential complications.  
Elevated white blood cell count
Absence of free air or diffuse peritoneal signs on initial imaging, suggesting contained inflammation rather than perforation
Clinical presentation includig abdominal pain, fever, constipation or diarrhea, and bloating
CT imaging shows colonic wall thickening (commonly >4 mm) in the sigmoid colon, with pericolic fat stranding
Diverticulitis rare:
Free air (pneumoperitoneum) on CT, indicating perforation
Dark-colored stools or gastrointestinal (GI) bleeding
Symptom improvement with antibiotics followed by relapse
Extensive diverticulosis of the colon observed on CT, without active inflammation
Widespread activation of immune cells throughout the body, potentially leading to shock, organ failure, or sepsis if left untreated.
Extraluminal gas localized to the left lower quadrant"""




pericarditis = """
Pericarditis general:
Positional chest pain improving with sitting up and leaning forward
Elevated inflammatory markers (e.g., CRP, ESR, WBC)
Electrocardiogram showin ST elevations or prolonged PR intervals.  
Evidence of pericardial effusion on imaging studies (echocardiogram, CT, or radiography)
Positional chest pain improving when sitting up and leaning forward
Pericardial friction rub upon auscultation.
Pericarditis rare:
Signs of cardiac tamponade (distended neck veins, hypotension).  
High ESR with marked leukocyrosis (e.g., Neutrophils ≥75%)
Radiographic evidence of water-bottle-shaped heart"""
# """



pneumonia = """
Pneumonia general:
Productive cough, often yellow, green, or blood-streaked sputum 
Low-grade to moderate fever (e.g., >100.4°F), commonly accompanied by chills and malaise
Shortness of breath that is progressive, typically gradual onset
Elevated white blood cell count with neutrophil predominance (neutrophilia), supporting infectious etiology
Ground-glass opacities or lobar/segmental consolidations on chest imaging (X-ray or CT)
Crackles (rales) or bronchial breath sounds heard on auscultation
Pneumonia rare:
Extensive lung consolidation involving multiple lobes, sometimes progressing to acute respiratory distress syndrome (ARDS)
Hypoxemia unresponsive to oxygen therapy, indicating severe pulmonary involvement
Weight loss, poor appetite, and night sweats, especially with chronic infections like tuberculosis or fungal pneumonia
Bibasilar atelectasis visible on imaging, secondary to obstruction or infection
Pleuritic chest pain that worsens with coughing or deep inspiration but does not improve with positional changes"""
# """



pulmonary_embolism = """
Pulmonary Embolism general:
Filling defects in pulmonary arteries on CT angiography (e.g., main, segmental, or subsegmental branches)
Tachycardia (rapid heart rate), typically defined as > 100 bpm
Elevated D-dimer levels
Hypoxia (low oxygen saturation, e.g., SpO₂ < 90%)
Signs of right ventricular strain on echocardiography
Direct visualization of thrombi via CT pulmonary anglography.  
Sudden onset of dyspnea
Pulmonary Embolism rare:
Wedge-shaped opacity consistent with pulmonary infarct
Extremely elevated D-dimer levels (>5000 ng/mL)
Electrocardiogram abnormalities (e.g., ST segment depression, T wave inversion)
Bilateral saddle emboli causing extensive occlusion
Pro-BNP elevation (e.g., 1147 ng/mL)
Presentation with cough, fever, and tachypnea
Elevated Troponin and BNP levels indicating cardiac strain"""



chest_reference = "\n".join([pericarditis, pneumonia, pulmonary_embolism])
abdomen_reference = "\n".join(
    [cholecystitis, appendicitis, pancreatitis, diverticulitis]
)