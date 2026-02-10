cholecystitis = """
Cholecystitis general:
Right upper quadrant abdominal pain and tenderness
Presence of multiple risk factors such as previous episodes of tumor flare-ups, recent changes in diabetic regimen leading to hypoglycemia, and use of Sorafenib therapy increasing the risk of thromboembolic events and subsequent development of ascending cholangitis
Positive Sonographic Murphy's sign
Imaging findings of dilated common bile ducts and choledocholithiasis on ultrasound abdomen
Elevated white blood cell count
Fever >38°C (100.4°F)
Elevated Liver enzymes (ALT, AST)
Cholecystitis rare:
Perforation of the gallbladder with abscess formation
Ischemic Gallbladder Necrosis due to Arterial Occlusion secondary to Pseudoaneurysms
Ascending Cholangitis secondary to Choledochoolithiasis
Presence of sludge within the gallbladder
Gallbladder wall edema and minimal thickening (<3mm)
Absence of choelithiasis does not rule out acute cholecystitis"""




appendicitis = """
Appendicitis general:
Abdominal pain localized to the right lower quadrant
Elevated white blood cell count
Presence of nausea and vomiting
Presence of periappendiceal fat stranding
Imaging findings of a dilated appendix
Tenderness to palpation over the right lower quadrant
Clinical presentation of nausea, vomiting, and fever
Appendicitis rare:
Presence of an appendicolith on imaging
Trace periappendical stranding
Non-filling of the appendix with mucosa hyperenhancement and dilatation of the appendiceal tip
Migration of abdominal pain from the peri-umblical area to the right lower quadrant
Free fluid tracking along the right paracolic gutter on computed tomography scan
Fluid-filled and dilated appendix measuring up to 13 mm in diameter
Dilation of the appendiceal tip, walls of the proximal and mid appendix are indistinct with surrounding fat standing, and adjacent cecal tip thickening"""


pancreatitis = """
Pancreatitis general:
Elevated serum lipase level (> 3 times upper limit of normal)
Nausea and vomiting
Imaging findings suggestive of pancreatitis (enlargement, necrosis, or inflammation)
Severe epigastric pain radiating to back, often worse after eating.
Elevated serum amylase level
Elevated white blood cell count indicating infection/inflammation.
Imaging evidence such as CT scan showing pancreatic inflammation/enlargement 
Pancreatitis rare:
History of recent heavy ethanol consumption,
Imaging showing a pancreatic mass with surrounding inflammation and necrosis,
Worsening symptoms despite previous treatment,
Presence of gallstones and a dilated common bile duct on ultrasound,
Splenic vein thrombosis,
Post-endoscopic retrograde cholangiopancreatography (ERCP)-induced pancreatitis indicated by recent ERCP procedure,
Elevated liver enzymes indicative of associated liver involvemen"""



diverticulitis = """
Diverticulitis general:
Surrounding fat stranding
Leukocytosis (Elevated White Blood Cell Count)
Presence of colonic diverticulosis on imaging studies
Fever
Thickening of the sigmoid colon
Diverticulitis rare:
Contained perforation indicated by extraluminal air
Small foci of extraluminal air adjacent to the sigmoid colon
Notable involvement of the sigmoid mesocolon with surrounding fat stranding
Micro-abscess formation
Phlegmonous fluid collection interposed between the sigmoid colon and bladder
Pneumoperitoneum due to perforation of diverticula"""





pericarditis = """
Pericarditis general:
Pleuritic chest pain that worsens with deep breathing
Elevated white blood cell count
Pericardial friction rub upon auscultation.  
Normal cardiac biomarkers (e.g., Troponin-T < 0.01 ng/mL)
Elevated inflammatory markers (such as CRP or ESR)
Widened mediastinum on portable chest radiograph
Pericarditis rare:
Positional variation in chest pain (improvement with leaning forward)
Uremic pericarditis due to end-stage renal disease (ESRD)
Low-grade fever (<101°F)
Bibasilar linear opacities consistent with discoid atelectasis and mild bronchial wall thickening on imaging studies
Presence of pericardial friction rub may be absent early in disease course
Radiation-free chest pain exacerbated by deep breathing and movement
Malignant pericarditis due to metastatic disease from non-Hodgkin's lymphoma (NHL)"""


pneumonia = """
Pneumonia general:
Clinical presentation consistent with respiratory infection (fever, cough, productive sputum).
Elevated white blood cell count
Radiographic evidence of new infiltrate(s), lobar or segmental pattern
Laboratory confirmation via Gram stain/culture of blood/sputum/tracheal aspirates or urinary antigen testing positive for Legionela pneumophilia or Streptococcus pneumoniae serotype
Patient's history of possible aspiration event with oral secretions.
Ground glass opacity in the right lower and middle lobe on CT scan indicating alveolar damage.
Pneumonia rare:
Recent history of dental procedure which may be considered as risk factor for aspiration pneumonia
Underlying condition of chronic lymphocytic leukemia (CLL) making patient susceptible to infections.
Bronchiectasis in the right lower lobe with peribronchial wall thickening.
Presence of risk factors for aspiration including history of recurrent aspirations, seizures, and recent enteral feeding tube placement.
Delirium secondary to pneumonia"""



pulmonary_embolism = """
Pulmonary Embolism general:
Sudden onset of dyspnea
Presence of risk factors such as recent surgery, immobility, and malignancy
Imaging findings of filling defects in the pulmonary arteries on CT scan
Elevated D-dimer level (>1000 ng/ml)
Pleuritic chest pain
Tachypnea
Pulmonary Embolism rare:
Extensive bilateral pulmonary emboli including a saddle embolus at the bifurcation of the main pulmonary artery
History of breast cancer with current radiation therapy and recent discontinuation of anticoagulation therapy
Heterozygosity for the prothrombin gene mutation indicating inherited thrombophilia
Wedge-shaped, peripheral-based consolidation consistent with pulmonary infarct
Near occlusive acute thrombus within the right main pulmonary artery with extension into the lobar, segmental and subsegmental branches of the right lung
Central filling defects in the right main, right middle, interlobar, and posterior basal segmental pulmonary arteries consistent with acute PE"""


chest_reference = "\n".join([pericarditis, pneumonia, pulmonary_embolism])
abdomen_reference = "\n".join(
    [cholecystitis, appendicitis, pancreatitis, diverticulitis]
)