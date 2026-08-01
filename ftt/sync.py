import trackio

import os

# Both keys and values must be strings
os.environ["HF_TOKEN"] = "hf_..."


import trackio.sqlite_storage as sqlite_storage
sqlite_storage.deserialize_values = lambda metrics: metrics  # keep "Infinity"/"NaN" as strings during sync

import trackio
trackio.sync(project="ner-extraction-conll2003", space_id="calcworks/sft-tutorial")
