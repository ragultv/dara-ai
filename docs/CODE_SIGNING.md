# Code signing the Windows installer (SignPath Foundation, free)

Why: the installer is unsigned, so Windows shows "Windows protected your PC" (SmartScreen) and the publisher as unknown. SignPath Foundation signs open-source projects' releases for free, with a certificate issued to SignPath Foundation, after the project is accepted.

**What signing does and does not do.** A signed installer shows a publisher name ("SignPath Foundation") instead of "Unknown publisher", and the signature proves the file came from this repository's automated build. SignPath's own pages do not promise that SmartScreen stops warning: Windows also looks at the reputation of the file and the certificate, which builds with downloads. Expect the warning to soften, not necessarily vanish on day one.

The release workflow is already prepared. It does nothing different until you finish the steps below, so merging it is safe.

## What the workflow does

`.github/workflows/build-release.yml`, job `installer`:

1. builds the installer as before;
2. when the build is going to be **published** (push to `main`, or a manual run with "release" ticked), or a manual run ticks **"sign"**, **and** the repository variable `SIGNPATH_ORGANIZATION_ID` is set, it uploads the installer as the workflow artifact `dara-installer-unsigned`, sends it to SignPath and waits for the signed file;
3. checks the signature on what came back (fails on "not signed" or a hash mismatch), then hashes, uploads and releases **the signed file**.

Pull requests never sign (they cannot read the secret). A published build with signing not set up yet still releases, with a visible warning in the run. A published build uses the policy `release-signing` (a person approves each release in SignPath); a manual "sign" trial uses `test-signing`.

What is signed is `.signpath/artifact-configuration.xml`: the installer `Dara_*_x64-setup.exe`. The files the installer unpacks (`dara.exe`, `engine\dara.exe`) are not signed separately, because Tauri builds the installer in one step.

## What you do

1. **Check the conditions** below, and make sure everyone on the signing team uses two-factor authentication on GitHub and SignPath.
2. **Apply** at <https://signpath.org/apply> (the project is Dara, the repository URL, MIT licence, the download page). SignPath may decline: they say they are under no obligation to accept a project. I could not read the application form, so follow what it asks.
3. **After acceptance**, in SignPath: create the project with the slug `dara` (any other slug works if you set `SIGNPATH_PROJECT_SLUG`), add the trusted build system **GitHub.com** to the organisation and link it to the project, create the artifact configuration from `.signpath/artifact-configuration.xml`, and make sure there are the signing policies `release-signing` (with an approver) and `test-signing`. Create an **API token** for a user who may submit to those policies.
4. **In the GitHub repository** (Settings > Secrets and variables > Actions):
   - secret `SIGNPATH_API_TOKEN`: the token;
   - variable `SIGNPATH_ORGANIZATION_ID`: the organisation id shown in SignPath;
   - optional variables, only if yours differ: `SIGNPATH_PROJECT_SLUG` (default `dara`), `SIGNPATH_SIGNING_POLICY_SLUG` (default `release-signing`), `SIGNPATH_ARTIFACT_CONFIGURATION_SLUG` (default `initial`, the name SignPath gives a new project's first configuration: check yours).
5. **Try it**: Actions > build-release > Run workflow, tick **sign** and leave release unticked. The run should end with a signed `dara-installer` artifact. Then do a real release; SignPath asks the approver to approve the signing request.
6. **Switch the wording on** (see the end of this file).

## The conditions, and where Dara stands

From SignPath Foundation's terms (<https://signpath.org/terms>); read them yourself, they can change.

| Condition | Dara |
| --- | --- |
| OSI-approved licence, no commercial dual licensing | MIT (`LICENSE`) |
| Actively maintained, already released in the form to sign | yes; releases exist as `b<number>` |
| Functionality described on the download page | the README and the site's Get Started page |
| No proprietary components | **check this**: the engine links NVIDIA's CUDA runtime statically (NVIDIA's redistributable licence), and the local build copies `cudart64_13.dll` next to it. The terms allow system libraries and unsigned upstream binaries, and mention "no proprietary or non-open-source component". Ask SignPath when you apply. |
| No malware, no hacking tools | yes |
| Privacy: say what leaves the computer, installer announces changes and can be uninstalled | the Privacy page; the installer is per-user with an uninstaller (README, "Code signing policy") |
| Product name equals the project name; same product version across builds | `productName` is "Dara"; the version is in `tauri.conf.json`: bump it per release |
| Team roles: authors, reviewers, approvers; every signing request approved | today one person (`ragultv`); add trusted people if you can, and list them in the README |
| "Code signing policy" section on the home page and the download page | README has it; add it to the site's Get Started page and the release notes when you switch on |
| Builds on GitHub-hosted runners | yes (`windows-2022`) |

## Wording to switch on after SignPath accepts the project

The README's "Code signing policy" section is truthful today (not signed yet). After acceptance, replace its first paragraph with:

> Free code signing provided by [SignPath.io](https://about.signpath.io), certificate by [SignPath Foundation](https://signpath.org).

(That exact line is what their terms ask for.) Do the same in `site/src/components/GetStarted.tsx` (and rebuild the site), put a "Code signing policy" heading with that line in `.github/release-notes.md`, and change the "not code-signed yet" notes in the README, the site and `docs/RELEASE.md`. Team roles must stay listed.
