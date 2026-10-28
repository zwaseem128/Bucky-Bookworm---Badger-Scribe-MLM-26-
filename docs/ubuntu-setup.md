# Setting up Ubuntu for kraken and Jupyter

This sets up an Ubuntu computer so it can run `notebooks/kade_line_segmentation.ipynb`, which
cuts the Kade pages into one image per line with kraken. kraken doesn't run on Windows, which is
why this is on Ubuntu. Everyone who follows these steps gets the same Python and the same package
versions, so a notebook that runs for one of us runs for all of us.

It takes about 15 minutes, most of it copying images. You need Ubuntu 22.04 or newer, about 3 GB
of free disk, and to have been added to the **Badger Scribe** shared drive with your @wisc.edu
account. Type each command into a terminal (Ctrl+Alt+T) and press Enter.

If your Ubuntu runs inside Windows (WSL), follow the same steps, except: skip step 1, and in
step 4 use the WSL box at the end of that step. Type every command into the **Ubuntu** app
(Start menu → Ubuntu), not PowerShell or Git Bash. Its prompt looks like `you@PC:~$`. If you see
`sudo: command not found` or `source: command not found`, you're in the wrong terminal.

## 1. Install the tools Ubuntu needs

WSL: skip this step. Ubuntu on WSL already has curl and git, and rclone is only needed outside WSL.

```
sudo apt update
sudo apt install -y curl git rclone
```

## 2. Install uv

uv installs Python and Python packages. It picks the Python version for us, so it doesn't matter
which one your Ubuntu came with (kraken needs Python 3.10 to 3.13).

```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Check it worked: `uv --version` prints a version number.

## 3. Make the Badger Scribe environment

An environment is a folder with its own Python and its own packages, kept apart from the rest of
the computer. Ours lives in `~/venvs/badger-scribe`, outside the repo, so it never ends up on
GitHub.

```
uv venv --python 3.12 ~/venvs/badger-scribe
source ~/venvs/badger-scribe/bin/activate
uv pip install --torch-backend cpu kraken jupyterlab ipykernel pandas openai pillow
python -m ipykernel install --user --name badger-scribe --display-name "Badger Scribe"
```

The third line takes a minute or two. `--torch-backend cpu` installs the 1.5 GB version of
PyTorch instead of the 6 GB one for NVIDIA GPUs. Kraken runs fine on a CPU. (If you have an NVIDIA
GPU and want to use it, use `--torch-backend auto` instead.)

The last line adds a kernel called **Badger Scribe** to Jupyter, which is how a notebook knows to
use this environment.

Check it worked:

```
python -c "import kraken, importlib.metadata as m; print('kraken', m.version('kraken'))"
```

It should print `kraken 7.1.1` (or newer).

## 4. Copy the images and train.csv from the shared drive

Google Drive for Desktop doesn't exist for Linux, so we use rclone to copy the files. Copying the
images once is faster and more reliable than reading each one over the network while kraken
works.

**Connect rclone to the shared drive (once):**

```
rclone config
```

It asks a series of questions. Answer:

| Question | Answer |
|---|---|
| `n) New remote` ... | `n` |
| `name>` | `badger` |
| `Storage>` | `drive` (Google Drive) |
| `client_id>` and `client_secret>` | press Enter (leave blank) |
| `scope>` | `1` (full access, so you can upload results later) |
| `service_account_file>` | press Enter |
| `Edit advanced config?` | `n` |
| `Use web browser to automatically authenticate rclone with remote?` | `y`. A browser opens; sign in with your **@wisc.edu** account and allow access |
| `Configure this as a Shared Drive (Team Drive)?` | `y`, then type the number next to **Badger Scribe** |
| `Keep this "badger" remote?` | `y`, then `q` to quit |

Check it worked: `rclone lsd badger:` lists the folders in the shared drive (`images`,
`output data`, ...). If it lists your own My Drive instead, run `rclone config` again, delete
`badger`, and answer `y` to the Shared Drive question.

**Copy the files:**

```
mkdir -p ~/badger-scribe-data
rclone copy badger:train.csv ~/badger-scribe-data
rclone copy badger:images ~/badger-scribe-data/images --include "kade_*" --progress
```

Drop `--include "kade_*"` to copy every image (about 300 MB). Running the same command again
only copies what's new.

The data stays in `~/badger-scribe-data`, outside the repo: the Kaggle data doesn't go on GitHub.

> **Ubuntu inside Windows (WSL):** skip rclone. Your Windows Google Drive for Desktop already has
> the files, and Ubuntu can ask Windows to copy them over. Run this in the Ubuntu terminal (no
> `sudo` or password needed):
>
> ```
> mkdir -p ~/badger-scribe-data/images
> DATA=$(wslpath -w ~/badger-scribe-data)
> powershell.exe -NoProfile -Command "Copy-Item 'G:\Shared drives\Badger Scribe\train.csv' '$DATA'; Copy-Item 'G:\Shared drives\Badger Scribe\images\kade_*' '$DATA\images'"
> ```
>
> It takes a minute or two and prints nothing when it works. Check with
> `ls ~/badger-scribe-data/images | wc -l`, which should print 436. If your Google Drive letter
> isn't G:, change both `G:`s in the last line. (`wslpath -w` turns the Ubuntu folder into a path
> Windows understands, like `\\wsl.localhost\Ubuntu\home\you\badger-scribe-data`.)

## 5. Tell the scripts where the data is

Lokesh's scripts and the notebook all read the data folder from `BADGER_DRIVE`. Save it once so
every new terminal has it:

```
echo 'export BADGER_DRIVE="$HOME/badger-scribe-data"' >> ~/.bashrc
source ~/.bashrc
```

## 6. Get the repo and open Jupyter

```
cd ~
git clone https://github.com/zwaseem128/Bucky-Bookworm---Badger-Scribe-MLM-26-.git badger-scribe
cd badger-scribe
git checkout lokesh/fine-tuning-setup
```

The notebooks live on the `lokesh/fine-tuning-setup` branch, not `main`, so the `checkout` line
matters: without it there is no `notebooks/` folder. (Skip the clone if you have it already; run
`git checkout lokesh/fine-tuning-setup` and then `git pull` instead.)

Then, every time you want to work:

```
source ~/venvs/badger-scribe/bin/activate
cd ~/badger-scribe
jupyter lab
```

A browser opens with Jupyter. (In WSL, copy the `http://localhost:8888/...` link it prints into
your Windows browser.) Open `notebooks/kade_line_segmentation.ipynb`, check the top right says
**Badger Scribe** (click it to change the kernel if not), and run the cells from top to bottom.

To stop Jupyter, press Ctrl+C twice in the terminal.

## If something goes wrong

| Problem | Fix |
|---|---|
| `sudo: command not found` (or `source`, `apt`) | You typed into PowerShell or Git Bash. Open the **Ubuntu** app and run it there. On WSL you can skip step 1 entirely. |
| No `notebooks/` folder, or the notebook isn't there | You're on `main`. Run `git checkout lokesh/fine-tuning-setup` in `~/badger-scribe`. |
| `uv: command not found` | Run `source $HOME/.local/bin/env`, or open a new terminal. |
| `ModuleNotFoundError: No module named 'kraken'` in the notebook | The kernel isn't **Badger Scribe**. Change it top right. If it isn't in the list, redo the last line of step 3. |
| `Can't find .../images` in the notebook | `BADGER_DRIVE` isn't set. Redo step 5, then restart Jupyter. |
| `rclone lsd badger:` shows your own files | The remote isn't pointed at the shared drive. See the check in step 4. |
| `No space left on device` | The environment needs about 1.5 GB and the images about 300 MB. Free some space and re-run the step that failed. |

To start over completely: `rm -rf ~/venvs/badger-scribe` and redo step 3. Nothing else is
affected.
