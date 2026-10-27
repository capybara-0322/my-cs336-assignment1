import torch
from cs336_basics.softmax import softmax


def decode(
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int,
        context_length: int,
        device: str,
        temperature: float=1.0,
        top_p: float | None = None,
        end_token: str = "<|endoftext|>"
):
    prompt_token_ids = tokenizer.encode(prompt)
    end_token_id = tokenizer.encode(end_token)
    assert len(end_token_id) == 1
    end_token_id = end_token_id[0]

    assert temperature >= 0

    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            input_ids = prompt_token_ids[-context_length:]
            inputs = torch.tensor(input_ids, dtype = torch.long, device=torch.device(device)).reshape(1,-1)
            output_logits = model(inputs)
            last_logits = output_logits[:,-1,:]

            # 转换成概率
            if temperature == 0:
                next_token_id = torch.argmax(last_logits).item()
            else:
                probs = softmax(last_logits/temperature, -1)

                if top_p is None:
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    accu_probs = torch.cumsum(sorted_probs, dim=-1)
                    prob_mask = accu_probs > top_p
                    prob_mask[..., 1:] = prob_mask[..., :-1].clone()
                    prob_mask[..., 0] = False
                    sorted_probs[prob_mask] = 0.0
                    sampled_index = torch.multinomial(sorted_probs, num_samples=1)
                    next_token_id = sorted_indices.gather(-1, sampled_index).item()
            if next_token_id == end_token_id:
                break
            prompt_token_ids.append(next_token_id)

    output_seq = tokenizer.decode(prompt_token_ids)
    return output_seq
                    



