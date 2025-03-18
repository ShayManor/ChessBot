import pandas as pd

from preproces_data import parse_eval


def chop_csv(input_file, output_file, max_bytes):
    with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
        bytes_written = 0
        # Read in chunks for memory efficiency
        while bytes_written < max_bytes:
            # Read a chunk of up to 4096 bytes or the remaining bytes
            chunk_size = min(4096, max_bytes - bytes_written)
            chunk = f_in.read(chunk_size)
            if not chunk:  # End of file reached
                break
            f_out.write(chunk)
            bytes_written += len(chunk)


if __name__ == "__main__":
    input_file = "data/random_evals.csv"
    output_file = "data/choppedTest.csv"
    with open(output_file, 'r') as f:
        lines = f.readlines()
    df = pd.read_csv(input_file, nrows=1*10**5)
    df['Eval'] = abs(df['Evaluation'].apply(parse_eval))
    df_filt = df[df['Eval'] <= 200.0]
    with open(output_file, 'w') as f:
        df_filt.to_csv(f)
        print(f"Chopped file written to {output_file}, total lines: {len(df_filt)}")
