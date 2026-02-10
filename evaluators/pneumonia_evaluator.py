from evaluators.pathology_evaluator import PathologyEvaluator
from utils.nlp import (
    procedure_checker,
    keyword_positive,
    treatment_alternative_procedure_checker,
)
from tools.utils import ADDITIONAL_LAB_TEST_MAPPING, INFLAMMATION_LAB_TESTS, MODALITY_SPECIAL_CASES_DICT
from icd.procedure_mappings import (
    PNEUMONIA_PROCEDURES_ICD10,
    PNEUMONIA_PROCEDURES_ICD9,
    PNEUMONIA_PROCEDURES_KEYWORDS,
    ALTERNATE_PNEUMONIA_KEYWORDS,
    ANTIBIOTIC_TREATMENT_KEYWORDS,
    VENTILATION_PROCEDURES_KEYWORDS,
    OXYGEN_THERAPY_PROCEDURES_KEYWORDS,
)

class PneumoniaEvaluator(PathologyEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathology = "pneumonia"
        self.alternative_pathology_names = [
            {
                "location": "lung",
                "modifiers": ["infect", "pneumonitis", "bacterial", "viral", "aspiration"],
            },
            {
                "location": "pneumonia",
                "modifiers": ["acute", "pneumonitis", "bacterial", "viral", "aspiration"],
            },
        ]
        self.gracious_alternative_pathology_names = [
            {
                "location": "respiratory",
                "modifiers": ["infection", "bacterial", "viral", "fungal"],
            },
        ]

        # Define required and neutral lab tests
        self.required_lab_tests = {"Inflammation": INFLAMMATION_LAB_TESTS+[50890],
                                   "Microbiology": [
                                        90234,  # 痰培养
                                        90201,  # 血培养
                                        50890,  # 降钙素原
                                    ],
                                    "Gas Analysis": [
                                        50818,  # 血气分析中的pH
                                        50820,  # PaCO2
                                        50821,  # PaO2
                                        50822,  # 血钾（重症肺炎相关）
                                    ],
            }
        for req_lab_test_name in self.required_lab_tests:
            self.answers["Correct Laboratory Tests"][req_lab_test_name] = []

        self.neutral_lab_tests = []
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Complete Blood Count (CBC)"]
        )
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Liver Function Panel (LFP)"]
        )
        self.neutral_lab_tests.extend(
            ADDITIONAL_LAB_TEST_MAPPING["Renal Function Panel (RFP)"]
        )
        self.neutral_lab_tests.extend(ADDITIONAL_LAB_TEST_MAPPING["Urinalysis"])
        self.neutral_lab_tests = [
            t
            for t in self.neutral_lab_tests
            if t not in self.required_lab_tests["Inflammation"]
            and t not in self.required_lab_tests["Microbiology"]
            and t not in self.required_lab_tests["Gas Analysis"]
        ]

        self.answers["Treatment Requested"] = {
            "Antibiotics": False,
            "Ventilation": False,
            "Oxygen Therapy": False,
            "Support": False,
        }
        self.answers["Treatment Required"] = {
            "Antibiotics": True,
            "Ventilation": False,
            "Oxygen Therapy": True,
            "Support": True,
        }

    def score_imaging(self, region: str, modality: str) -> None:
        # Region must be chest
        if region == "Chest":
            # CT is preferred but Chest X-ray accepted
            if modality == "CT":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 2
                return True
            if modality == "Radiograph":
                if self.scores["Imaging"] == 0:
                    self.scores["Imaging"] = 1
                return True
        return False

    def check_antibiotics_time_order(self) -> bool:
        doc = self.nlp(self.answers["Treatment"].lower())

        antibiotics = False
        after_treatment = False
        for token in doc:
            if token.text == "antibiotics":
                antibiotics = True

            if (
                token.text
                in [
                    "treatment",
                    "curation",
                    "treating",
                    "course",
                    "case",
                    "resolution",
                    "therapy",
                    "intervention",
                    "management",
                ]
                and token.head.text.lower() == "after"
            ):
                after_treatment = True

        if antibiotics and after_treatment:
            return True

    def score_treatment(self) -> None:
        ### ANTIBIOTICS ###
        if keyword_positive(self.answers["Treatment"], "antibiotics"):
            self.answers["Treatment Requested"]["Antibiotics"] = True
        
        ### VENTILATION ###
        if procedure_checker(VENTILATION_PROCEDURES_KEYWORDS, self.procedures_discharge):
            self.answers["Treatment Requested"]["Ventilation"] = True

        ### OXYGEN THERAPY ###
        if procedure_checker(OXYGEN_THERAPY_PROCEDURES_KEYWORDS, self.procedures_discharge):
            self.answers["Treatment Requested"]["Oxygen Therapy"] = True

        ### SUPPORT ###
        if (
            keyword_positive(self.answers["Treatment"], "fluid")
            or keyword_positive(self.answers["Treatment"], "pain")
            or keyword_positive(self.answers["Treatment"], "respiratory")
        ):
            self.answers["Treatment Requested"]["Support"] = True

        ### ANTIBIOTICS (ALTERNATE TREATMENT) ###
        if treatment_alternative_procedure_checker(
            ALTERNATE_PNEUMONIA_KEYWORDS, self.answers["Treatment"]
        ):
            self.answers["Treatment Requested"]["Antibiotics"] = True

