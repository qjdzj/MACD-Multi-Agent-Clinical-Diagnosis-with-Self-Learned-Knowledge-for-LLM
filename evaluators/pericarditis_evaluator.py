from evaluators.pathology_evaluator import PathologyEvaluator
from utils.nlp import (
    keyword_positive,
    procedure_checker,
    treatment_alternative_procedure_checker,
)
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING, INFLAMMATION_LAB_TESTS
from icd.procedure_mappings import (
    PERICARDIOCENTESIS_PROCEDURES_ICD9,
    PERICARDIOCENTESIS_PROCEDURES_ICD10,
    COLCHICINE_TREATMENT_KEYWORDS,
    PERICARDIECTOMY_PROCEDURES_KEYWORDS
)

class PericarditisEvaluator(PathologyEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathology = "pericarditis"
        self.alternative_pathology_names = [
            {
            "location": "pericard",
            "modifiers": ["inflammation", "inflammatory disease"]
            },
            {
            "location": "pericardial",
            "modifiers": ["inflammation", "inflammatory change", "pericardial inflammation"]
            },
            {
            "location": "heart sac",
            "modifiers": ["inflammation"]
            },
            {
            "location": "cardiac membrane",
            "modifiers": ["inflammation"]
            }
        ]
        self.gracious_alternative_pathology_names = [
            {
            "location": "pericard",
            "modifiers": ["effusion", "fluid accumulation", "thickening", "fibrosis", "adhesion", "calcification"]
            },
            {
            "location": "pericardial",
            "modifiers": ["effusion", "thickening", "fluid", "fibrosis", "adhesions"]
            }
        ]

        # Laboratory test configuration
        self.required_lab_tests = {
            "Inflammation": INFLAMMATION_LAB_TESTS + [
                50889,  # CRP
                51288   # ESR
            ],
            "Cardiac Damage": [
                51003,  # Troponin T
                50963,  # BNP
            ],
        }
        for req_lab_test_name in self.required_lab_tests:
            self.answers["Correct Laboratory Tests"][req_lab_test_name] = []

        self.neutral_lab_tests = []
        self.neutral_lab_tests.extend(ADDITIONAL_LAB_TEST_MAPPING["Complete Blood Count (CBC)"])
        self.neutral_lab_tests.extend(ADDITIONAL_LAB_TEST_MAPPING["Renal Function Panel (RFP)"])
        self.neutral_lab_tests = [
            t for t in self.neutral_lab_tests
            if t not in self.required_lab_tests["Inflammation"]
            and t not in self.required_lab_tests["Cardiac Damage"]
        ]

        # Treatment evaluation configuration
        self.answers["Treatment Requested"] = {
            "AntiInflammatory": False,
            "Colchicine": False,
            "Corticosteroids": False,
            "Pericardiocentesis": False,
            "Pericardiectomy": False,
            "Antibiotics": False
        }
        self.answers["Treatment Required"] = {
            "AntiInflammatory": True,
            "Colchicine": True,
            "Corticosteroids": False,
            "Pericardiocentesis": False,
            "Pericardiectomy": False,
            "Antibiotics": False
        }

    def score_imaging(self, region: str, modality: str) -> None:
        """Score according to pericarditis imaging guidelines"""
        if region == "Chest":
            # Echocardiogram is the gold standard
            if modality == "Echocardiogram":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            # CT/MRI for complex cases
            if modality in ["CT", "MRI"]:
                if self.scores["Imaging"] < 2:
                    self.scores["Imaging"] = 1
                return True
            # X-ray has low sensitivity
            if modality == "Radiograph":
                if self.scores["Imaging"] < 1:
                    self.scores["Imaging"] = 0.5
                return True
        return False

    def score_treatment(self) -> None:
        """Treatment scoring logic"""
        # Anti-inflammatory drug treatment
        if keyword_positive(self.answers["Treatment"], "nsaid|ibuprofen|aspirin"):
            self.answers["Treatment Requested"]["AntiInflammatory"] = True
        
        # Colchicine treatment
        if procedure_checker(COLCHICINE_TREATMENT_KEYWORDS, [self.answers["Treatment"]]):
            self.answers["Treatment Requested"]["Colchicine"] = True

        # Pericardiocentesis
        if (procedure_checker(PERICARDIOCENTESIS_PROCEDURES_ICD9, self.procedures_icd9)
        or procedure_checker(PERICARDIOCENTESIS_PROCEDURES_ICD10, self.procedures_icd10)
        ):
            self.answers["Treatment Requested"]["Pericardiocentesis"] = True

        # Pericardiectomy
        if procedure_checker(PERICARDIECTOMY_PROCEDURES_KEYWORDS, self.procedures_discharge):
            self.answers["Treatment Requested"]["Pericardiectomy"] = True

        # Antibiotic treatment (infectious pericarditis)
        if keyword_positive(self.answers["Treatment"], "antibiotic") and self._is_infectious_case():
            self.answers["Treatment Requested"]["Antibiotics"] = True

        # Adjust necessary treatment requirements
        if self._has_cardiac_tamponade():
            self.answers["Treatment Required"]["Pericardiocentesis"] = True
        if self._is_recurrent_case():
            self.answers["Treatment Required"]["Corticosteroids"] = True
            self.answers["Treatment Required"]["Pericardiectomy"] = True
