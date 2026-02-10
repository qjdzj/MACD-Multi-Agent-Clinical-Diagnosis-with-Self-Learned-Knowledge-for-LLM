#!/usr/bin/env bash
set -euo pipefail

PATIENT_LIST_DIR=
PAST_BASE_DIR=
LOCAL_LOGGING_DIR=
GPU_ID="3"  

# Default models if none provided explicitly after GPU_ID
MODELS=("Llama3Instruct70B")
# If additional args remain after the first four, treat them as model names
if [ $# -gt 0 ]; then
  MODELS=("$@")
fi

declare -A PAST_DIRS=()

slug_to_pathology() {
  local slug="$1"
  # Replace underscores with spaces for Hydra pathology override
  echo "${slug//_/ }"
}

resolve_past_dir_for_slug() {
  local slug="$1"
  if [ -n "${PAST_DIRS[$slug]:-}" ]; then
    echo "${PAST_DIRS[$slug]}"
    return 0
  fi
  if [ -d "${PAST_BASE_DIR}/${slug}" ]; then
    echo "${PAST_BASE_DIR}/${slug}"
    return 0
  fi
  if [ -d "${PAST_BASE_DIR}/round1/${slug}" ]; then
    echo "${PAST_BASE_DIR}/round1/${slug}"
    return 0
  fi
  echo "${PAST_BASE_DIR}"
}

# Discover all patient list files
mapfile -t PATIENT_FILES < <(find "$PATIENT_LIST_DIR" -type f -name "*incosistent_patient_id_list_copy.pkl" | sort)
if [ ${#PATIENT_FILES[@]} -eq 0 ]; then
  echo "No *incosistent_patient_id_list.pkl files found under $PATIENT_LIST_DIR" >&2
  exit 1
fi

echo "Found ${#PATIENT_FILES[@]} patient list files under $PATIENT_LIST_DIR"
echo "Output will be saved to: $LOCAL_LOGGING_DIR"

declare -A FILE_FOR_SLUG=()
for f in "${PATIENT_FILES[@]}"; do
  base=$(basename "$f")
  # Extract disease slug before _round1_incosistent_patient_id_list.pkl
  slug=${base%_round2_incosistent_patient_id_list_copy.pkl}
  FILE_FOR_SLUG["$slug"]="$f"
done

# Iterate over models and diseases
for model in "${MODELS[@]}"; do

  for slug in "pulmonary embolism"; do
    if [ -z "${FILE_FOR_SLUG[$slug]:-}" ]; then
      echo "[WARN] Missing patient_list file for disease slug '$slug' under $PATIENT_LIST_DIR. Skipping." >&2
      continue
    fi
    patient_list_path="${FILE_FOR_SLUG[$slug]}"
    pathology="$(slug_to_pathology "$slug")"
    past_dir="$(resolve_past_dir_for_slug "$slug")"

    echo "- Disease: $pathology"
    echo "  patient_list_path: $patient_list_path"
    echo "  past_diagnosis_dir: $past_dir"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python /data2/kunzhang/LLM-MDM/infer.py \
      model="${model}" \
      prompt_template=DIAGSUM_WITH_PAST \
      pathology="$pathology" \
      patient_list_path="$patient_list_path" \
      past_diagnosis_dir="$past_dir" \
      local_logging_dir="$LOCAL_LOGGING_DIR"
  done
done 