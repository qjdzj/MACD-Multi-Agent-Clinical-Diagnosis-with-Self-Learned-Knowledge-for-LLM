import re
from evaluators.pathology_evaluator import PathologyEvaluator
from utils.nlp import (
    keyword_positive,
    procedure_checker,
    treatment_alternative_procedure_checker,
)
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING, INFLAMMATION_LAB_TESTS
from icd.procedure_mappings import (
    OXYGEN_THERAPY_PROCEDURES_KEYWORDS,
)

class PulmonaryEmbolismEvaluator(PathologyEvaluator):
    """Evaluate the trajectory according to clinical diagnosis guidelines of pulmonary embolism."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                # 增加病理名称的模糊匹配模式
        self._diagnosis_patterns = [
            re.compile(r'\b(?:pulmonary\s+embolism|PE)\b', re.IGNORECASE),  # 主模式
            re.compile(r'\b(?:pulmonary|PE)\b.*?\b(?:embolus|thrombus)\b', re.IGNORECASE)  # 替代模式
        ]
        self.pathology = "pulmonary embolism"
        self.alternative_pathology_names = [
            {
                "location": "pulmonary",
                "modifiers": ["embolism", "embolus", "thrombus"],
            },
            {
                "location": "PE",
                "modifiers": []
            },
        ]
        self.gracious_alternative_pathology_names = []

        # Define required laboratory tests
        self.required_lab_tests = {
            "Inflammation": INFLAMMATION_LAB_TESTS,
            "Coagulation": [
                50915,  # D-dimer
                50885,  # Fibrinogen
                51274,  # PT
                51275,  # PTT
                51675,  # "INR(PT)",
                51237,  # "INR(PT)",
            ],
        }
        for req_lab_test_name in self.required_lab_tests:
            self.answers["Correct Laboratory Tests"][req_lab_test_name] = []

        # Define neutral lab tests
        self.neutral_lab_tests = []
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Complete Blood Count (CBC)"]
        )
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Renal Function Panel (RFP)"]
        )
        self.neutral_lab_tests = [
            t for t in self.neutral_lab_tests
            if t not in self.required_lab_tests["Coagulation"]
            and t not in self.required_lab_tests["Inflammation"]
        ]

        # Initialize treatment tracking
        self.answers["Treatment Requested"] = {
            "Anticoagulation": False,
            "Thrombolysis": False,
            "Embolectomy": False,
            "Oxygen Therapy": False,
            "Support": False,
        }
        self.answers["Treatment Required"] = {
            "Anticoagulation": True,
            "Thrombolysis": False,  # For high-risk cases
            "Embolectomy": False,  # For surgical cases
            "Oxygen Therapy": True,
            "Support": True,
        }

    def score_diagnosis(self) -> None:
        # """通过正则表达式优先匹配完整诊断术语"""
        raw_diagnosis = self.answers["Diagnosis"]
        
        # 优先检查完整术语匹配
        for pattern in self._diagnosis_patterns:
            if pattern.search(raw_diagnosis):
                self.scores["Diagnosis"] = 1
                self.scores["Gracious Diagnosis"] = 1
                return
        
        # 保底使用父类逻辑（单词级模糊匹配）
        super().score_diagnosis()

    def score_imaging(self, region: str, modality: str) -> None:
        """Score imaging based on PE diagnostic guidelines"""
        if region == "Chest":
            # Preferred modalities
            if modality == "CT Pulmonary Angiography":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            if modality == "V/Q Scan":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
            # Alternative modalities
            if modality == "Echocardiogram":
                if self.scores["Imaging"] < 1:
                    self.scores["Imaging"] = 0.5
                return True
        return False

    def score_treatment(self) -> None:
        """Evaluate treatment appropriateness"""
        # OXYGEN THERAPY
        if procedure_checker(OXYGEN_THERAPY_PROCEDURES_KEYWORDS, self.procedures_discharge):
            self.answers["Treatment Requested"]["Oxygen Therapy"] = True

        # SUPPORTIVE CARE
        if (
            keyword_positive(self.answers["Treatment"], "fluid")
            or keyword_positive(self.answers["Treatment"], "analgesi")
            or keyword_positive(self.answers["Treatment"], "hemodynamic")
        ):
            self.answers["Treatment Requested"]["Support"] = True

        # Determine required treatments based on severity
        if self._is_high_risk_case():
            self.answers["Treatment Required"]["Thrombolysis"] = True
            self.answers["Treatment Required"]["Embolectomy"] = True

    # def _is_high_risk_case(self) -> bool:
    #     """Determine if case meets high-risk criteria"""
    #     # Implementation would check for hemodynamic instability, biomarkers, etc.
    #     return (
    #         keyword_positive(self.answers["Presentation"], "hypotension")
    #         or keyword_positive(self.answers["Presentation"], "cardiac arrest")
    #         or keyword_positive(self.answers["Lab Results"], "elevated troponin")
    #     )

