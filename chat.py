import os
import torch
from train import BigramLanguageModel, encode, decode, device, checkpoint_file

checkpoint = torch.load(checkpoint_file, map_location=device)
history = "history.txt"

max_new_tokens = 200

model = BigramLanguageModel().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

conversation = ""
if os.path.exists(history):
    with open(history, "r", encoding="utf-8") as f:
        conversation = f.read()
    print("--- Previous Chat History Loaded ---")
    print(conversation, end="")
    print("-----------------------------------")

if __name__ == "__main__":

    while True:

        user_input = input("<ongthee> ")

        if user_input.lower() == "exit":
            break

        conversation += f"<ongthee> {user_input}\n"

        with open(history, "a", encoding="utf-8") as f:
            f.write(f"<ongthee> {user_input}\n")

        context = torch.tensor([encode(conversation)],dtype=torch.long,device=device)

        output = model.generate(context, max_new_tokens)

        generated = decode(output[0].tolist())

        response = generated[len(conversation):]

        if not (response.startswith("<ongthee>")):

            if "<ongthee>" in response:
                response = response.split("<ongthee>")[0].strip()

            print(response.lstrip())

            with open(history, "a", encoding="utf-8") as f:
                f.write(response.lstrip() + "\n")

        conversation += response.lstrip() + "\n"