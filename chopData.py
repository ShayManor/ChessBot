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
    print(f"Chopped file written to {output_file}, total bytes: {bytes_written}")


if __name__ == "__main__":
    input_file = "data/chessData.csv"
    output_file = "data/choppedData.csv"
    max_bytes = int(150 * 1024 * 1024)
    chop_csv(input_file, output_file, max_bytes)
    with open(output_file, 'r') as f:
        lines = f.readlines()
    with open(output_file, 'w') as f:
        # f.writelines(lines[98 * len(lines) // 100:-1])
        f.writelines(lines[:-1])
