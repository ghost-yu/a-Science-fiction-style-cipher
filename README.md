# a-Science-fiction-style-cipher

A method of encryption that uses a large language model (LLM) to generate coherent science-fiction text that hides a ciphertext inside natural prose.

## Public protocol (shared openly)

1. **Pre-shared secret**  
   The sender and trusted recipient share a secret passphrase out of band.

2. **Message preparation**  
   - Convert plaintext to UTF-8 bytes.
   - Encrypt with a stream/keyed cipher derived from `SHA-256(passphrase + nonce)`.
   - Base64-encode the ciphertext.

3. **Fiction carrier generation**  
   - Prompt the LLM to produce a short science-fiction story.
   - Embed the Base64 ciphertext by taking the **first letter of each sentence** in order.
   - Include the nonce in clear text in the story metadata (for example: `TIME-JUMP-ID:<nonce>`).

4. **Public artifact format**  
   Publish only:
   - The generated story
   - The nonce (`TIME-JUMP-ID`)
   - The embedding rule identifier (`SENTENCE_INITIALS_V1`)

## Trusted recipient decryption

1. Read sentence initials from the story in order (`SENTENCE_INITIALS_V1`).
2. Reconstruct the Base64 ciphertext.
3. Derive the same key from `SHA-256(passphrase + nonce)`.
4. Decrypt and decode UTF-8 plaintext.

## Notes

- Anyone can read the story and nonce, but only the trusted person with the passphrase can decrypt.
- Coherent fiction provides plausible cover text while preserving a deterministic extraction rule.
