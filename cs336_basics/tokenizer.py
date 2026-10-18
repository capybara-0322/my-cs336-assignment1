from typing import Iterable, Iterator
import pickle
import regex as re


class Tokenizer:

    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    BYTES_TOKENS = tuple(bytes([i]) for i in range(256))


    def __init__(self,
        vocab:dict[int,bytes], 
        merges:list[tuple[bytes,bytes]], 
        special_tokens:list[str] | None=None
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_size = len(vocab)

        self.bytes_to_id = {token : token_id for token_id, token in vocab.items()}  # construct bytes_to_id
    

        
        return None


    @classmethod
    def from_files(cls,
        vocab_filepath:str, 
        merges_filepath:str,
        special_tokens:list[str] | None=None
        ):
        
        with open(vocab_filepath,"rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath,"rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)
    


    def word_to_byte_tuple(self, token: str):
        """Convert a token to a tuple of its byte representation."""
        encode = token.encode("utf-8")
        return tuple(self.BYTES_TOKENS[b] for b in encode)


    def apply_merge(self, pre_token:tuple[bytes, ...]) -> list[bytes]:
        """ Merge one pre_token
        
        """


        tokens = list(pre_token)
        for left, right in self.merges:
            result = []
            i = 0

            while i < len(tokens):
                if(i+1 < len(tokens) and tokens[i] == left and tokens[i+1] == right):    # match left and right
                    result.append(left+right)
                    i+=2    # processed 2 bytes
                else:       # didn't match 
                    result.append(tokens[i])
                    i += 1

            tokens = result

        return result
            

    

    
    def encode(self, text: str) -> list[int]:

        # split text with special tokens
        if self.special_tokens:
            escaped_tokens = [re.escape(tok) for tok in self.special_tokens]
            pattern = "(" + "|".join(sorted(escaped_tokens, key=len, reverse=True)) + ")"
            segments = [seg for seg in re.split(pattern,text) if seg]
        else:
            segments = [text]



        result_ids = []
        compiled_pattern = re.compile(self.GPT2_PAT)
        for segment in segments:
            if self.special_tokens is not None and segment in self.special_tokens:
                result_ids.append(self.bytes_to_id[segment.encode("utf-8")])
            else:
                pre_token_data_list = []    # list[tuple] tuple是每个单词的bytes元组

                # pre_tokenizer 
                for match in compiled_pattern.finditer(segment):
                    token_merged_bytes = self.apply_merge(self.word_to_byte_tuple(match.group()))
                    for token_bytes in token_merged_bytes:
                        result_ids.append(self.bytes_to_id[token_bytes])

        return result_ids

            

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode a stream of text chunks into token IDs."""
        compiled_pattern = re.compile(self.GPT2_PAT)

        special_tokens = self.special_tokens or []
        if special_tokens:
            escaped_tokens = [
                re.escape(token)
                for token in sorted(special_tokens, key=len, reverse=True)
            ]
            special_pattern = re.compile(
                "(" + "|".join(escaped_tokens) + ")"
            )
        else:
            special_pattern = None

        def encode_segment(segment: str) -> Iterator[int]:
            """Encode a segment without special tokens."""
            for match in compiled_pattern.finditer(segment):
                token_bytes = self.apply_merge(
                    self.word_to_byte_tuple(match.group())
                )
                for token in token_bytes:
                    yield self.bytes_to_id[token]

        buffer = ""

        for chunk in iterable:
            buffer += chunk

            # Find the longest suffix that might start a special token.
            suffix_length = 0

            for token in special_tokens:
                for length in range(1, min(len(token), len(buffer)) + 1):
                    if buffer.endswith(token[:length]):
                        suffix_length = max(suffix_length, length)

            safe_end = len(buffer) - suffix_length
            safe_text = buffer[:safe_end]

            if special_pattern is not None:
                segments = special_pattern.split(safe_text)
            else:
                segments = [safe_text]

            # Keep the last ordinary segment for future chunks.
            remaining = ""

            for index, segment in enumerate(segments):
                if not segment:
                    continue

                if segment in special_tokens:
                    yield self.bytes_to_id[segment.encode("utf-8")]
                    continue

                matches = list(compiled_pattern.finditer(segment))

                if index == len(segments) - 1:
                    if matches:
                        for match in matches[:-1]:
                            yield from encode_segment(match.group())
                        remaining = matches[-1].group()
                    else:
                        remaining = segment
                else:
                    yield from encode_segment(segment)

            buffer = remaining + buffer[safe_end:]

        # Flush remaining text.
        if special_pattern is not None:
            segments = special_pattern.split(buffer)
        else:
            segments = [buffer]

        for segment in segments:
            if not segment:
                continue

            if segment in special_tokens:
                yield self.bytes_to_id[segment.encode("utf-8")]
            else:
                yield from encode_segment(segment)
            


    def decode(self, ids: list[int]) -> str:
        bytes_list = []
        for token_id in ids:
            bytes_list.append(self.vocab[token_id])
        bytes_seq = b"".join(bytes_list)
        string_out = bytes_seq.decode("utf-8",errors="replace")
        return string_out
        





