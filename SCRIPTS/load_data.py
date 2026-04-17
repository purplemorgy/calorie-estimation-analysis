from io import BytesIO
from datasets import Dataset, Image
from mlcroissant import Dataset as JsonldDataset
from sklearn.model_selection import train_test_split

JSONLD_URL = "https://huggingface.co/api/datasets/mmathys/food-nutrients/croissant"


def _decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {
            k.split("/", 1)[-1] if "/" in k else k: _decode_value(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_decode_value(v) for v in value]
    return value


def _encode_image(value):
    if isinstance(value, bytes):
        return value
    if hasattr(value, "save"):
        try:
            if hasattr(value, "load"):
                value.load()
            if getattr(value, "mode", None) not in ("RGB", "RGBA"):
                value = value.convert("RGB")
            buffer = BytesIO()
            value.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as exc:
            fileobj = getattr(value, "fp", None)
            if fileobj is not None:
                try:
                    fileobj.seek(0)
                    return fileobj.read()
                except Exception:
                    pass
            raise ValueError("Unable to encode image object to PNG bytes") from exc
    return value


def _normalize_record(record, encode_images=False):
    normalized = {}
    for key, value in record.items():
        normalized_key = key.split("/", 1)[1] if key.startswith("default/") else key
        decoded = _decode_value(value)
        if normalized_key == "image" and encode_images:
            decoded = _encode_image(decoded)
        normalized[normalized_key] = decoded
    return normalized


def _load_raw_records(encode_images=False):
    jsonld_dataset = JsonldDataset(jsonld=JSONLD_URL)
    records = [_normalize_record(record, encode_images=encode_images) for record in iter(jsonld_dataset.records("default"))]
    if len(records) == 0:
        raise ValueError("Loaded dataset contains no records.")

    for record in records:
        if "split" in record and isinstance(record["split"], bytes):
            record["split"] = record["split"].decode("utf-8", errors="replace")
    return records


def load_food_dataset():
    records = _load_raw_records(encode_images=True)
    dataset = Dataset.from_list(records)
    dataset = dataset.cast_column("image", Image())
    return dataset


def get_train_val_test_splits(test_size=0.2, val_size=0.1, seed=42):
    records = _load_raw_records(encode_images=False)
    label_name = "total_calories"

    if len(records) == 0 or label_name not in records[0]:
        raise ValueError(
            f"Dataset does not contain the expected label column '{label_name}'. "
            "Please use a dataset that includes calorie labels."
        )

    if any(record.get("split") == "train" for record in records) and any(record.get("split") == "test" for record in records):
        train_val_records = [record for record in records if record.get("split") == "train"]
        test_records = [record for record in records if record.get("split") == "test"]
    else:
        train_val_records, test_records = train_test_split(
            records,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )

    if val_size <= 0:
        return train_val_records, None, test_records

    val_ratio = val_size / (1.0 - test_size)
    train_records, val_records = train_test_split(
        train_val_records,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )
    return train_records, val_records, test_records


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_train_val_test_splits()
    print("train", len(train_ds), "val", len(val_ds) if val_ds else None, "test", len(test_ds))
    print(list(train_ds[0].keys()))
