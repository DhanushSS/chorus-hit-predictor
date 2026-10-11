"""Bounded CSV parsing with explicit, actionable schema and cell errors."""
import csv
import io
from collections import Counter
import numpy as np
import pandas as pd

MAX_CSV_BYTES = 5 * 1024 * 1024


def parse_csv(payload, columns, *, reorder=False):
    if not payload: raise ValueError('The CSV is empty. Download the sample or feature template.')
    if len(payload) > MAX_CSV_BYTES: raise ValueError('CSV limit is 5 MB; supply 1–100 records.')
    try: text = payload.decode('utf-8-sig')
    except UnicodeDecodeError as error: raise ValueError('Save the CSV with UTF-8 encoding.') from error
    if '\x00' in text: raise ValueError('CSV contains binary/null bytes; upload a text CSV.')
    reader = csv.reader(io.StringIO(text, newline=''), strict=True)
    try:
        header = next(reader)
        duplicates = [n for n, count in Counter(header).items() if count > 1]
        if duplicates: raise ValueError('Duplicate columns: ' + ', '.join(duplicates[:8]))
        if len(header) == 518:
            raise ValueError('This looks like a legacy 518-feature chorus CSV. Use the 436 whole-recording HSP-S template; the features cannot be renamed or truncated.')
        missing = [n for n in columns if n not in header]; extra = [n for n in header if n not in columns]
        if missing or extra:
            raise ValueError(f'Schema mismatch. Missing {len(missing)}: {", ".join(missing[:6]) or "none"}. Extra {len(extra)}: {", ".join(extra[:6]) or "none"}. Use the template.')
        if header != columns and not reorder:
            raise ValueError('Correct unique columns, wrong order. Enable explicit column reordering or use the template order.')
        records = []; errors = []
        for row_number, row in enumerate(reader, 1):
            if row_number > 100: raise ValueError('Maximum 100 records per CSV; split the file into smaller batches.')
            if len(row) != len(header): raise ValueError(f'Record {row_number}: expected {len(header)} cells, found {len(row)}. Check delimiters/quoting.')
            values = []
            for column, cell in zip(header, row):
                if len(cell) > 128: raise ValueError(f'Record {row_number}, {column}: numeric cell exceeds 128 characters.')
                try: value = float(cell) if cell.strip() else np.nan
                except ValueError:
                    errors.append(f'record {row_number}, {column}: not numeric'); value = np.nan
                if np.isinf(value): errors.append(f'record {row_number}, {column}: infinity is not allowed')
                values.append(value)
            if np.isnan(values).all(): errors.append(f'record {row_number}: all feature values are missing')
            records.append(values)
        if errors: raise ValueError('Invalid cells: ' + '; '.join(errors[:8]))
        if not records: raise ValueError('CSV has a header but no records. Upload 1–100 rows.')
        return pd.DataFrame(records, columns=header).loc[:, columns]
    except (csv.Error, StopIteration) as error:
        raise ValueError('CSV could not be parsed. Check UTF-8 encoding, comma separators and balanced quotes.') from error
