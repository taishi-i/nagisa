import os
import re


# e.g., <NE:ORGANIZATION:京都大学>
NE_PATTERN = re.compile(r"<NE:([A-Z]+):([^>]+)>")


def find_ne_span(words, ne, phrase_start, phrase_end):
    # An NE tag is attached to the basic phrase that contains the last word of the NE,
    # so search for the words that end in the basic phrase and match the NE string.
    for end in range(phrase_start + 1, phrase_end + 1):
        for start in range(end):
            if "".join(words[start:end]) == ne:
                return start, end
    return None


def convert_to_iob2_tags(words, phrase_starts, nes):
    tags = ["O"] * len(words)
    phrase_ends = phrase_starts[1:] + [len(words)]

    for netype, ne, phrase_id in nes:
        span = find_ne_span(words, ne, phrase_starts[phrase_id], phrase_ends[phrase_id])

        # Skip an NE that is a part of a word (e.g., 英 in 英語).
        if span is None:
            continue

        start, end = span
        tags[start] = "B-"+netype
        for i in range(start + 1, end):
            tags[i] = "I-"+netype

    return tags


def load_kwdlc(filename):
    data = []

    words = []
    phrase_starts = []
    nes = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("+ "):
                # A basic phrase line may have an NE tag.
                phrase_starts.append(len(words))
                match = NE_PATTERN.search(line)
                if match:
                    netype, ne = match.groups()
                    nes.append([netype, ne, len(phrase_starts) - 1])

            elif line.startswith("# ") or line.startswith("* "):
                continue

            elif line == "EOS":
                tags = convert_to_iob2_tags(words, phrase_starts, nes)
                data.append([words, tags])

                # reset
                words = []
                phrase_starts = []
                nes = []

            else:
                # The first field of a morpheme line is the surface form.
                surface = line.split(" ")[0]
                words.append(surface)

    return data


def write_file(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        for words, tags in data:
            for word, tag in zip(words, tags):
                f.write("\t".join([word, tag])+"\n")
            f.write("EOS\n")


def main():
    kwdlc_dir = "./KWDLC"

    for split in ["train", "dev", "test"]:
        # Use the official train/dev/test split of the documents.
        fn_id = os.path.join(kwdlc_dir, "id", "split_for_pas", split+".id")
        with open(fn_id, "r", encoding="utf-8") as f:
            doc_ids = [line.strip() for line in f if line.strip()]

        data = []
        for doc_id in doc_ids:
            # e.g., ./KWDLC/knp/w201106-00000/w201106-0000060050.knp
            fn_knp = os.path.join(kwdlc_dir, "knp", doc_id[:13], doc_id+".knp")
            data += load_kwdlc(fn_knp)

        fn_out = "kwdlc."+split
        write_file(fn_out, data)


if __name__ == "__main__":
    main()
