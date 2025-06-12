def parse_fasta(file_content, max_len=20):
    """Parses FASTA formatted text and returns a list of (header, sequence) tuples.

    Args:
        file_content (str): The content of a FASTA file as a string.
        max_len (int, optional): Maximum length for the header (default is 20).

    Returns:
        list of tuple: A list of tuples, each containing a unique header (str) and its sequence (str).
    """
    sequences = []
    header = None
    seq_parts = []

    for line in file_content.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith('>'):
            if header:
                sequences.append((header, "".join(seq_parts)))
            header = line[1:].strip().replace(" ", "_")[:max_len]
            seq_parts = []
        elif header:
            seq_parts.append(line.replace(" ", "").upper())

    if header:
        sequences.append((header, "".join(seq_parts)))

    final = []
    used = set()
    for h, s in sequences:
        uh = h
        count = 1
        while uh in used:
            uh = f"{h[:max_len - 2]}_{count}"
            count += 1
        used.add(uh)
        final.append((uh, s))

    return final
