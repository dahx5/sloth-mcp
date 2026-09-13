# sloth-mcp

**The more it clicks, the better it gets.**

sloth lets Claude use your Mac — native apps and websites in the browser alike. Not by
looking at the screen before every click, but by remembering what it has already seen.

![Five calendar events from one plan](media/calendar.gif)

Five events in Calendar: 25 clicks and keystrokes, sent by Claude as a single plan. An
agent that looks before every step would have sent 25 screenshots, about 35,000 tokens
of pictures. Claude got none — just a text report of what happened at each step.

## Looking is expensive. Remembering is cheap.

Every window sloth sees, it writes down: what is in it and where each button leads.
The next time Claude needs that app, it doesn't look. It reads the map — a few hundred
tokens instead of a 1,400-token screenshot — and plans the whole route from it. On a
walk through System Settings, 16 of 20 windows were recognised from memory; the text on
screen had to be read only four times.

So sloth gets better with use in the most literal way: the apps you open most are the
ones it no longer needs to look at.

Claude sees a screenshot only when reality stops matching the plan — a button that
isn't there, a dialog nobody expected. Then it gets the picture along with a log of
everything that happened so far, and carries on from the failure instead of starting
over.

## If it's on the screen, sloth can click it

sloth reads pixels, not the Accessibility API, so the browser is just another app to
it. Native apps like Calendar, Finder or System Settings; websites in Safari or Chrome;
Electron apps; a web drawing tool it has never seen before:

![Excalidraw, thirteen actions from one plan](media/excalidraw.gif)

## Try it

You need macOS, [Claude Desktop](https://claude.ai/download) and
[uv](https://docs.astral.sh/uv/). sloth itself needs no API keys and runs entirely on
your Mac.

```bash
git clone https://github.com/dahx5/sloth-mcp.git
cd sloth-mcp
uv sync
uv run choto service install
```

(The commands are called `choto` — the project's name before it became a sloth.)

Then add it to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sloth": {
      "command": "/opt/homebrew/bin/uv",
      "args": ["run", "--project", "/path/to/sloth-mcp", "choto-bridge"]
    }
  }
}
```

Restart Claude Desktop and allow **Screen Recording** and **Accessibility** for
`Choto.app` when macOS asks.

To stop it mid-task, push the mouse into the top-left corner of the screen.

## What it can't do

- Anything other than macOS, or more than one display.
- Buttons that are only an icon, out of the box. sloth finds things by their text; for
  icons it needs an icon detector you supply yourself:
  `uv run choto model install --icon-detector /path/to/icon_detect.mlpackage`
  (not included; the weights are AGPL-3.0).
- Promise you a reply. This was built for my own Mac and shared as is. Issues are
  welcome.

MIT License.
