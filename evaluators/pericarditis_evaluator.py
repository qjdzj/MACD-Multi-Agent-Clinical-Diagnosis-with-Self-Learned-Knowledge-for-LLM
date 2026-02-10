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

        # 实验室检查配置
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

        # 治疗评估配置
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
        """根据心包炎影像学指南评分"""
        if region == "Chest":
            # 超声心动图为金标准
            if modality == "Echocardiogram":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            # CT/MRI用于复杂病例
            if modality in ["CT", "MRI"]:
                if self.scores["Imaging"] < 2:
                    self.scores["Imaging"] = 1
                return True
            # X光检查灵敏度低
            if modality == "Radiograph":
                if self.scores["Imaging"] < 1:
                    self.scores["Imaging"] = 0.5
                return True
        return False

    def score_treatment(self) -> None:
        """治疗评分逻辑"""
        # 抗炎药物治疗
        if keyword_positive(self.answers["Treatment"], "nsaid|ibuprofen|aspirin"):
            self.answers["Treatment Requested"]["AntiInflammatory"] = True
        
        # 秋水仙碱治疗
        if procedure_checker(COLCHICINE_TREATMENT_KEYWORDS, [self.answers["Treatment"]]):
            self.answers["Treatment Requested"]["Colchicine"] = True

        # 心包穿刺术
        if (procedure_checker(PERICARDIOCENTESIS_PROCEDURES_ICD9, self.procedures_icd9)
        or procedure_checker(PERICARDIOCENTESIS_PROCEDURES_ICD10, self.procedures_icd10)
        ):
            self.answers["Treatment Requested"]["Pericardiocentesis"] = True

        # 心包切除术
        if procedure_checker(PERICARDIECTOMY_PROCEDURES_KEYWORDS, self.procedures_discharge):
            self.answers["Treatment Requested"]["Pericardiectomy"] = True

        # 抗生素治疗（感染性心包炎）
        if keyword_positive(self.answers["Treatment"], "antibiotic") and self._is_infectious_case():
            self.answers["Treatment Requested"]["Antibiotics"] = True

        # 调整必要治疗需求
        if self._has_cardiac_tamponade():
            self.answers["Treatment Required"]["Pericardiocentesis"] = True
        if self._is_recurrent_case():
            self.answers["Treatment Required"]["Corticosteroids"] = True
            self.answers["Treatment Required"]["Pericardiectomy"] = True
