fn main() -> anyhow::Result<()> {
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        anyhow::bail!("build with --features cuda and/or --features metal");
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
        run()
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn run() -> anyhow::Result<()> {
    use anyhow::{bail, Context};
    use llama3_goldy::checkpoint::Checkpoint;
    use llama3_goldy::generate::generate;
    use llama3_goldy::model::Model;
    use llama3_goldy::tokenizer::Tokenizer;
    use std::path::PathBuf;

    fn print_usage() {
        eprintln!(
            "Usage: llama3-goldy [prompt] [--checkpoint PATH] [--tokenizer PATH] [-n TOKENS]\n\
             Defaults: models/stories15M.bin, models/tokenizer.bin, 50 tokens,\n\
             prompt \"I have a dream\""
        );
    }

    let mut checkpoint = PathBuf::from("models/stories15M.bin");
    let mut tokenizer_path = PathBuf::from("models/tokenizer.bin");
    let mut prompt = "I have a dream".to_string();
    let mut n_tokens: u32 = 50;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            "--checkpoint" | "-c" => {
                checkpoint = PathBuf::from(args.next().context("missing --checkpoint path")?);
            }
            "--tokenizer" | "-z" => {
                tokenizer_path = PathBuf::from(args.next().context("missing --tokenizer path")?);
            }
            "-n" | "--n" => {
                n_tokens = args.next().context("missing -n value")?.parse()?;
            }
            flag if flag.starts_with('-') => bail!("unknown flag {flag}"),
            other => prompt = other.to_string(),
        }
    }

    eprintln!("loading {}", checkpoint.display());
    let ckpt = Checkpoint::read_path(&checkpoint)?;
    eprintln!(
        "config dim={} hidden={} layers={} heads={} kv_heads={} vocab={} seq={}",
        ckpt.config.dim,
        ckpt.config.hidden_dim,
        ckpt.config.n_layers,
        ckpt.config.n_heads,
        ckpt.config.n_kv_heads,
        ckpt.config.vocab_size,
        ckpt.config.max_seq_len
    );
    let tokenizer = Tokenizer::from_path(&tokenizer_path, ckpt.config.vocab_size())?;
    let mut model = Model::load(&ckpt)?;
    let out = generate(&mut model, &tokenizer, &prompt, n_tokens)?;
    print!("{}", out.text);
    println!();
    std::io::Write::flush(&mut std::io::stdout())?;
    eprintln!(
        "Token count: {}, {:.0} tokens/s, worker records={}",
        out.tokens.len() + 1,
        out.tokens_per_second,
        out.worker_records
    );
    Ok(())
}
