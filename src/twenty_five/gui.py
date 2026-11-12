"""Local browser interface for playing Twenty-Five against project policies."""

from __future__ import annotations

import argparse
import base64
from functools import lru_cache
import gzip
import json
import random
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import rules
from .cards import Card
from .engine import InvalidAction, Phase, Transition, TwentyFiveGame
from .observations import PlayerView
from .policies import POLICIES, FrozenCheckpointPolicy, Policy, make_policy


_RANK_LABELS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
_SUIT_NAMES = ("hearts", "diamonds", "clubs", "spades")
_SUIT_SYMBOLS = ("♥", "♦", "♣", "♠")

# The complete browser client is embedded so this module can be copied by itself.
_EMBEDDED_ASSETS = {
    "index.html": "H4sIAAAAAAACCo1XzW7cNhC++ylY9tICVTYx0KIHrQAnceoATmrY20OPXHF2xZoSVZLa7d7yBj0GfYD2FXpv38RP0hlSlCXtGvXB8HKGnP/5ZpR/IU3pDy2wyte6OMvpH9Oi2S45NJwIIGRxxlhegxesrIR14Je885vse/7IaEQNS75TsG+N9ZyVpvHQ4MW9kr5aStipErJw+IapRnkldOZKoWH56kiMr6CGrDTa2JGkL1/K8835d/G2V15Dsdoj55C9UzvIF5FETK2ae2ZBL7nzBw2uAkCTKgubJV9E0ovSOXJvEf3L10Yewls6g2WlFs6hJaZdCxtUIk+qXWKsrWhkT0eOa0UzYWW1sPe8OP82XxBvuIgiitx5a5rt1PqelrtaaF1cGwwNu3jPvFhr4gZqvqDn0ZbRz5FZaG8mSq9M42bGKbnkpm1NgzpfC7kFnt4karYO5GJm8brz3sT30uwbbYR8HUiDgF87Bfi6J1I1YRD6k1SOPJDFndgB5qTV4pAvIveUigb2P2ANzDS0VmE8D6d1FB9hz7b4aCp3iFDMMdjiLJSZUEOmRNtmWBxapwQ7CLFLfBJKZdjVDR9nMPFdaSxkmDnV8mB9INzFs7BKZFqsqQzviO54SuAJUSHP2d6KdtA0vbAB7aOS8Ovx0vG1zMNvvrPAxwUzuhosBeGvxQHsEGOioL1Emtl52tYSSwbsxJDpNWs66gNs69mlcUmGS9cUJV7c0m/2alp/0ye9aBKaoSolo7VPPwhlCwgztlfyNhzYP3+zm5OajkI28912dYuqHRXysV8pvBj+BuQbYeUQ3xpRLyuJUpzQMUBDaPTV7U8fboauj9AQ5ZL63pOHP/5kVyCsd4/occL4J/yJ0lR5f2FB8Ef3kJKJQBoX8JvOWsw3C/xTdTUlTI6zop/1GMUxwyyAfqLwKZQZIjpMa37ILvHfGs8n9yUSTtRFfhyLUXBba+oW310jxCmkhTp/+PTXEN7ZS8pOfFgJB1eqobewReCmNDsMICKeUA4kw4nAKrWtNP55kC9Sbp8ZwlHOKpo7ydd4GCfqZ9NZFsjFLAl93BM4Dsd4Fg57aYJ72mxPgh7SMwJUjNAsY6l6P1ys3lwN1VudFysKJKO5tFMe4R9Jc/eGdDqAHg9YGOhL/tFYlMSoix27B2gZ7gfKMrrJcDLshAfWNV5pYrBa+LJi0Eh3NMomAUkBhR0W9jU6mxwMBJfCisMZE2y08lM8zRchZHHA0FCJ40XiXmO2vSe+a9+G8whkkZbFS2nobNC/xwfv8MQZAlxlCLrGN6d5iKz/ScXd5Yqtri7Z6uL19eU4JXcegYPhwoWzM0TsZFZGg7nUxsHMG4W72RNzfwIe9JQX/34+GvxHCUlTDIOQba2SY7eCtOImjCiHMKKxgmP7RdL7pu1i1wcW/ogv5hJCi9C0G8ug8zMFJCNxmUWvcEdLC9TEqER8rlVxLo0lxLGV3ps2oOZO6A7DTCumqXFkhv/5InKfo2XdbWPnfGWx0KmlSM/XuSI9QwdGrX1Om65e06KAE2zJX3KG8S6hMhqH75Lf9pbMNE5T244WuZ0oD1ljQjuF7nz4/Hvfp+xHbOAIXy5A5gYVZbR1MmvWtEtGYHVeHFLnI5S280UydQg6NitP161r5eMWENAiCJwvjlR+sbFjk6XW7qemEW40buLJGoIqNMx3boQTuStxGfTM2RI/PXDffPELIouEDa6j6HRg0ldI/PzAHgxfYf8BPvqB05YNAAA=",
    "styles.css": "H4sICKccmWoCA3N0eWxlcy5jc3MArVpbj6vIEX6fX4HO6Eh2Al5oLgb7JcpDopWSPGQTKXlsQ2OTweAAnsuu9r+nqi/QTWOPZxUdzZyhobuq6/LVpXvXte3g/PLkOJ5XNS875znYkiAo9nzkfB1YAWNJsg0TOXahF9bBWBmXAUu1Ma9g7AIvWMZIEYsXJasHGPLLmEbhNOQVtENa/iEsw0CMH9saSRUJTaODGOo48cOWFGEmRvoTLdq3neM7QXp5d2IffnXHA10FxHXCwHVI7DqbIFnvn359evqd84tzaN+9vvq5ao47+LsrgE8Y2ju/Pp2Gc+3CWPEBn52rxjux6ngCdgPf/44fiFdA90y7Y9UAVWQib+sW9v9KuxUX2RoHDzR/OXbttSl28OQ4HS0qWntH/J81wyqvurxmDh1g8e9O/N0VXJMgcoMkdbexuwnitesMHW36C+1gDmylY+e1y9erq4bRbloviPyCHV3QQlrGrIA/GGNpkXNmyrYZvJKeq/pj5/zYDKxznWvl9bC017OuKl2n/+gHdvaulet49HKpmSdGXOePQOrlrzT/iT//CZZynW8/sWPLnH/++A1mjqugiA/XYWgbGGU1y+HLqrlcwZw4Czt4OsGHA5cl/xDe5NeuR/ld2go5w3dPm6G9HGjHZa2UsE0u77iZCy0KrjzfyWt6vqxQ8a4Tvb65Tgzf8B0XVX+pKey2rBmfRevq2HgV7KDfOTnjlGD4P9d+qMoPLwf2GHIIss6Zd2DDG2MNV6QyEWD3DKYA9tW3dVVIfYVuGLokA23N1S4/iH2XRCn8EHeTkvGjomsvXlnVA7rOob52q0Dw/uvT5gA6L0Ay5iYWt+AcKXhYQC7cgMVMD6zzhYvurSqG086J+GslR/E0rn3sqmLvwN+wbUs80rTfTvBmb2xOWPvouuu9FNTOIaOExDfoxmtNkmixV6ASo08Zlvln1oJbUTQetCbxElyVwaI+Ms0H3pQ9+L5Y9V0DgQiIozQkBgRuEroRaIfoku2Hrm3AV9Tjmda1Lu5D3eYvkzzl99KIH+I2QG5rNoAYPbQobq8bP2BnfVlJ18APDrBrY7WArzaw98HjaFC2HRji9QIIm9OeLRAKIiQkQMoDV+JK4ZThwaP5ULVN/yUD8+X89nJpG3jjHWhxZI8wH+JMyUonNBfLxS5dBS8+PIUZm/9eKzZMjwUDxFQ4YRtQhsuMaIAsOkG8aCajbfrcCiy6BqEvGr1ufTGyENnWF6RrvmF9fxiJtKU1nN8vRJRxB7fxZ2vT2IGC6aFmCCYtWscAut5EZD+CbtMOHlhh+8aKBZXsTu0rxgpdQGJsxv1zQCCexzYDSx9LwSLMQ6QB8TF0gwmuIPKugtj3EdUx8q73U7x16HVoNa0TlLYMAxFOCDEMhBAR1vy3EQsE1OFvD2wcxgYGwF9fz03PqZ7p+yohgm7ZrR2ScnPiHkDQg9B4jvSsZsksQbLt8933edtB7By66iIyBS2NSKIF5OWrp5qTjJGGCPZv8NuxC6PDStgILsi6HqIgpAxyJylRGxHYJzjLaVdwxh7x/U+i48S7HpYzCcH7ub8KEFmIkLGrfjZbooWKO9buZ/NNbRDWXhlP8DhZw4lEEDLctWp6NoBF4T+kY3xoLv3RXucGX5bl7Kv7eD4PLjpE+l/GdxlHdOJfjlEK0sUiYLD5S/9wNOI+zDliCCJvHb3IxA3xxsNn5EXmNzzZEXKXBise1kLSCKjcJi9tX2FsQusGawd17g0HEo4ehdxF4+T15DoJZvywDOJMWaNeT1VR6Kmb8CRpRs9JHOUhsWwzSr87P8CH328B/nqea+jmgyD0nB22EYC99iKOVSjwXfy3CWUawiMIKvzasdnG6QEYvSI+8nU4rEzQTYBzmSd77BVcskfpN8xMPT3A8CNTEAEm482rhTDmxYJeXPhOiFK1XdIn8Rrf4S6l08GqHkcc0PF91uXXNzdZs3JQYkY18mRFPH7g4+QP/E+EwJUHmSNUKTFGBitH1QA5i0e/4ug24drPEFQL9g5OMO1naF9Y81nWbMYhkcg8lkkvJb8WFMZumLhkC4VCsphQ28pJiF3gLTr9LB2yspa5qYLSR9lsIFvgNmJIaoazz+U2p0k2X1y5h2A92rrEJ27gx1NSxCmcrmfazNe3HFGCs6IYBiQJonERr4HQvJDH6+BFrPwwRYFwI9FaCcAwWvyz7wsbRoj1kB9ESGUSAkFncdQKmcrm9Gw8ktA7LqvbnciYx9ZDpqUGwlu8eCG4agFXD5vYBbDq0puwEAlYeN5mJAkDEARPIZ4PJPKjktvhiAGYEaLyT5jnjBohfpD6xX4WKsN8mxA9acNGzcIe0lE5QluZpawM9WEIIxSiVIFHSPszTAXvEw4svBh98RGk0fLTOIAXMRGxx0SgBcBBgQlkPrOB2oXXTaO5XZFx2UjCy2mUHbTtgjTFROJO5gF8I8deUb1WBYd7KQPDRsMFXVoAN0+vhu56vnhSEA/WojflNIlEITRmof6U5JjUeKYGZc1sVKVQcwDRNYpWd2vFyRGWgdqy7RuFu736V5O7IJX7BlutZM4/qs7QXRQZVfSSKqN52i4NfaFKXeTvwTDEWzf+PAytzW1sOl7MGqQ7NpoU5LAepDPUyEumXHIE1e1kFzhFCmgJMT7DBadrBxySrPCH9Xq2NiRLB1bfoKBQhb4ru1ZpkQAmDBg5rfMV1sLO71FFNzKjf60EVFl2dhsn4rVATzQzjAh2eSh9YEmcknPemJ1k6wsYrzG/wugvXo8WthXl4tSRWWzrqIASJMEhyEzzs1vfPGxhSQa/kjJkh7XdGcTTAX5EYJgXWbtWIah/4Kd38qtv/6jOrHf+xt6cv7eQv3wbXdFGDK4kKcFRd3gw0bsao3JAJvz8SXSsNPXcdQFDjxug83I/IZKdSpDoWGRtsiVftdbur9XwydrKycyJh+roycn3a4cHEmudXJgtkNvk7RmEOUzwF2e6taaRMlbIw/qqV+9Hc00WAHHLv1gqw26SV6rQMXp7j18lX31CcneCJlddCZlKkHCWd6GNAKKxbQpccN8QOoaKtOFp4ewELIoWct3fEKO1XrYVUEfSY39eG7rdo5/P+y1h0ia02AaxOupWaaHWKlrN5AyLy76cLNkF1VisGceFydqgv3mjFWb5885VyrIsi8xlR+M9iZMn4/CTpF+pcwyj4I7FmkLv02OcR2jgnQ3ZtfEgVojO7jjyMXZyRr5mUWpWFQXcRKdeSZwuR8l/rzIeQRdX3ZVV1w9Y1tQWBf/GnE3NjrTWW+zBLcq+pU6e8RAyj0vYxxW6JgutyVs8jO32ZeoeEhszFi+AqGnzkyA/8W1+eF3M4kMWb2/JUDt1UOcMBSvptR4EEvEWet0eZ4bGW/57ngiNDT+Z97yeHA8MMeaKm1kit7Gi6lguQonokOsWlz3Wd36s62x1tWWEMISISgsnL50WkD0P2DzskRaVWW/ccK1Z130Rdr9wTK0kM32s+hE6W4tAKE3w8zomwTrGLt1nNE7kM7BW5Rzh5Zy/mF8894wVf5EZ9qfta+sQQp6gjBdW1prpIFZtFwBbRWIR/HETPN0NTPyanVUJV8eZfd61dX2AJFZGiOFUNdOCN6qXcaFUNGL4/yKB+3TXxErzgk00bkvgm9Y1KrIizNOJo93uwABOxIGvsMpv37SiYNbQ9bhBYZgUzSyxy60WBheEOg+Cz4fsEFG61Ad9LmmZlmTiT9Rac/VrNamywiT257PU3iDfEKMIT9qG7zUxxAT+ck5cPyU202mDhaPoWd6feq+ilgqOlg++EW4hn75iL4EKyNW7WOKwc6ouPQfPVta6Cxin5xb4pZb1yeNqowEotWWCJFICVFg4IhEN3YlpUIa8MTNTB5+4dUnqEuJuklQStu/WxCriG7LgJZiWCxPl1uL9/wmheXzqB9oNS2s/CLOBv9wLtZF3KztIMzoj1MI2ETdmpqRwNjLbZhXsULuvoNnBQ1cXFpBIJJMmEolDQFCGh5WeLu67J/ZB2eGPqizUrZJpofrRmBDcORV48EhWtFJt8htwN3RwvgfBOhB0foB8dfa5ujGnDanLc+pkgF9D1IsQs7USBA94o7rWYh8VFGlB8u1yV769DqgzrVKYM74r2/za2+yL8ceP5acsc+lCpLqk8krzD69ph4euIBlHd8FU7BrrgCqbJSAWJM0bQoY+JvGH4yUr2ptBvKze8YKN1tZTuRcRh5S3u4zyjGGsLXyr9SBUIhi3Ly09k20YEi190C5KmZpOZ4mscVKvS1RsVOtobUg/HaoSfxLCpj+1bzfqEbE/X9sb94inP5wZYJezwjpgEjSvXLAxqd8ZuoMMuBR8rJcaWl0RCjzVqw8SSdCDWeJCglmcqCtwvy4xmKQTf+rq6ljxz9NAIm+JSWLaZUDwHvOC3exOnA6NyhFnEjHJpLKYkx8al5MWiu8vXDVa490sFVaXRcbvaYxGpprFS2eX6ciheYqnHC0h39UO+GUC/ZR/O6nNOD1Wpx2xcVI1yWLWS1CUjF52lsy7g0kyLiBbJVYuoMd7VWlVjUBPQiYb0+Ldp2b8SFARKkQD/R//+MANwy8AAA==",
    "app.js": "H4sIAAAAAAACCs0Z23LbNvbdX4GgaUw2Eu0kzc7Wsp1J4+wmO0nqib0PO5mMBZGQxJoiVBK0rVH5Xfu+X7bnHAC8iZLSnT6sX0wB545zBUKV5prJRC5kqnN2xtYHjOWhyuSVzuLlCYtUWOBe8Fshs9UVQIZaZR7/rgbi/oDlUugPYiWzXRgOBhCAS6aKNPogJjLZgVMDIZdIikRm+3AaUIaTLrJURm9EFu3AqoGQk86KxXIfoxrI8sni8PZ1JsVOHAuDXJaZWiz1DmgDYKgv5yKX7+J0J7yDIR1Aowul9+gMEIb+XKS77IPbSFXewe4HNdsB6kAM3VzKvcdcwSAHtVyqFKB+FtFM7kBqwRlekbpPEyWinwutVbrLR1qAFjkWyU69cqmL5QVBoZxTlS32Qf8NYOzxJej6+a7DMxDv02WhXUztCScHC/SdNb7BYBUHEyl7Q6nBBQ9qzzlW1LUS+U7vw33uH5Sjg4NEapZroSWkoLRIkhGtTIp8BQtTkeTSrCSA8hbd630EG8eAOS3SUMcqZTIPxVK+u/74wbsTSSF9ymWZRCdnmKjSmd0IMgm2DqV39OXJ6fkh/3o0G7BwLjJ2ds68NX/CT/gTsViO+ICf4nei8fMcP2f0eYif3734Cb4P+SF8/1Yo2Cj9L0jmqz86KBuShZBUSC78AE4Q0yLUTrGWnOPTfCkAA/TMzzh6BIg9RDz2eI3/glAlKivxlyXzinH7ydkJ47zkTGSxGCYYT2f88bphGKJAG37Jz4EtYy2GmUhv+fkmCq775ekRwp63MPIi1ohBYPlqMVFJL9wkng13wIIo9mvcth05ibeQeS5m1lSuWgW0F2j5oN+oVMMSmNRCjjbhSJAPMXyJKPJ4Plf33B+RT+vreCFVoT3PRxfYipjJhbqTDnfAnr88PjZnLfJVGrJK6kyCs4PcS6HnmNBwkapraVQIqexmMoeAzNHnxb2INZtKHc4t0nouRUQJY82tesPr1VJyOGOxXCZxKJDq0a855K9ywIIgsHxK0sqwWIoVprmKg2MZIJpHgPGUeY+qdXXrMz3P1D1L5T17m2UQqpZIIPEX+/13xq/nkmkxSSSwKZKIpUqTW0OIwsZcaGeBwJjYercl1D5hzGOXKo/xh4c/Bi5XDti8WIj0CtbaZktA9Ts0G8GzYQ3Hnjpkn33vPmtziHSWIN5HMHFw+Z4dseeA4VUUj2rsHyqgH9jzhg7rhxP28hiwaDtUuUdEEeHHFwO2au7mcdrYfV62Fc9kCgd8hX1U7nVcu+6uglyvEhmAk15maikzvfL4cGjF5AOTNYOZWMjAyT7aQilOU5lhUIMFDBqdYbAQS8/govePT6P4rgpvxHb5x8AEcX5D9sbMs1KFzTqw3xAlLDJQT984smdn1rIBndiTJ+xRA1rLbBGnIkGKkMfgJFwqszkKJIJ0shBJ0k5OlmYKRCg9EcRprjOVzs4rgUkJ3Dbrp0eGXDOLkZrUm+W8RjQLpWnsOquk0zOUmGTNeSuVIYexH/yqwAM493sPHgyxee6uR/7Gw1qTfWys2yBChGZMNQzf5y5ujc6UoMiBHNWJIGVZz+Gj4u2ChY4yJAy04ussE6tgCj2st05kOtPzExMXcNhOKmwrbyCJpCDbX3zIYyYFH57GGzQ5nFx8flhbtaxLBspbldCm/6K39XouffzJvmsxqzpMoXsG4fpwAjLY8wgeyu9Hw+GqubQqv7eu3inIyHGo1a1MG27ZVONfyKwp3lP2rOGJ/QQxXviOSGqZw/SAG2Z4BWfUJGzAhuEcBsLzCyvBobGE/fV4Ta5Rmpg2MYJHV+4OlGuMN69ZASgCq7hoHRpttau/m7haIUWrFEyk+oDFwOrB78ZUs9YgXGALTjdkwOw9iRhK0OZqM7j216M+qhulyZF7AFJbatOLH5uAqxZgq0y9eNkEzJQWNqcM/wrwZCWAerk95MiyVDIaAUB0JLi8I1hGcjZK5FTDGgaEVkv4ojgwDWLllYHpmmHYxhzfdLmak2l2+fmliZGNSPh2Z3sH+aiblDFHtZyHc+obGz5AIDAQvhXQwFG57DjShAZNwK3GoRBcUsu3honHJ3YSNYY1vwINDR/ys5utPTLCJ3AMABjvnhSo65czk6Tow6aocYtiFOdYYrBbfNTAgYYPB7EWKLQjrzXYH35CP1zPG9CQjB1D/F12uQ+Y+USBSYgBNY9GqHLcVr9p8/H/x6DSthhMEjSN4nQgQVaPh9CY33JXxlBLvFIywpkUY1Vsexd09OB+b+ZxEnmGtm+ctc9HieVG6+BuXXo6B9rKKdvR50abZ1bBnPg/uAVBy451zY6bw8qeDsd5OswN0Ph3eQvtDZ/5r4I4Qn86doOHBX4ExaUx2vs9WuVhppLkWi2B8rbddzKezTXSbt8TGC59pvTc9N1sxUftDm3UrkOjVqYYdc6kVXrq28vOlDr+jDvttoOATURWBBpXmV0KF6Ys/+ff7LKvXGPya9OqLzjbqQO6l9jljWZfQ+A3jUyylVwrTLcSwdAsTyfZ+VYIG2kdPtUla9cCbToIdmMpsM2tL8dfYZz/J4RZ9kbk0us0OQQT5BC80nvmd0Qwt7Ad9gbZbLWh3R1sL4Kp5AhCHRXP1IRjXsSJmg4BJioWzpUCgDhn8ziKZMqwh8ab12xV3fMFmD239KFXWmTQXNAIj5sMxvZc3OFoDjTp9mtF+PwDJWNknDORSZaJOIcCAK4NrGfzBONJRgHfOPoL1bwY0WoG7YPH8Y4BylB7LO300xjrG+0T5IQeXTqTrLsk3m7YSE6K2Q0CghKvMEjgN11atl0CV0rss+4gWkBgfwzG8PqtOb7qR0cUfpnFd3hriVHH2+K2rqe7vnuXVxSrAzUFs5MAWvfUrQrdI+0uVB1rajW3uMyFhYajZ/IB7xX/cfXLJ+sr5Cqf6TNnBcBB9y2mgA4OJdlC6HDOtMLnDC0B0zoteRXvux6ryqKg3yYHYznAPsO3XSUqYy+AsffrpF16blnZFsvdHLs7LnP5xo/EMj4yLMAj1wup5yrCQ/vl6hoWJipanZCaQU7Xw/F05a0NfOmXvmtxTZ2gcgw6oa4e3YSB2PZ6kn5WxXHEcMKZomUTkLB9i90pHmWrJk3jJGleC9i53vT9QOJTsZiALHWqMTuBudauS7CNOLoQO6tH7i46Rp7BpZrsVxdvMOA868YegDbzfN9sb7FxiL8xzxe+6TTMxSQjVngjjVvQYFyZ4bzRrBvAbl/RJ/BZU8laczesdkuoweqprS2lDjuSQsmK1IKff6b/lXiHIO2fof+lSYh/xAJNbWqHMKs+O60O8JWzBLi71WLU42q6WHp19D3acCx7oxwYDX0bbhtg+xzjJ2uSesbetImZLJ+y59R51r+cSttMUx70COTs0z99N8Jsu3t1S1Rvbu/ptd2WabehkkNL0a9xc3YhsOBWrvxu42237KPNXgepJGvrUS0Di42iWsFy4yQb5uwZc2BqmUlIopUtQYytL33Qh/wdzsA9tu4dm9ZN9xw1vJ/eXgN8fvmoIpFQBt3FOEygj3IvtnvZdvkQtocD27aiul8RjKu+zobdg4ur+wBqKd2HBPNMTvEIqGjZsmu0q5jjg3MPy7yYLGBeHTBTZasJj6ZDGtuWGf2/kFNRJPqPVE84uW8vnfa+sXrp3l2u7OvOjXnm3l6b6N0Z/+oX7i3eXj9o9+ZLR8e8YfdEAO3Xlb/71rzNP0b/Y4+AfA7a5qajAE+Ftir1qrc79CN3SO4JreaJN1rE0rBA8D6OwOy/awlaEu0jAAA=",
}


@lru_cache(maxsize=None)
def _asset_bytes(name: str) -> bytes:
    return gzip.decompress(base64.b64decode(_EMBEDDED_ASSETS[name]))


def card_payload(card: Card) -> dict[str, Any]:
    """Return display-safe card data using the stable action index."""
    suit = int(card.suit)
    return {
        "index": card.index,
        "rank": _RANK_LABELS[int(card.rank)],
        "suit": _SUIT_NAMES[suit],
        "symbol": _SUIT_SYMBOLS[suit],
        "color": "red" if card.suit.is_red else "black",
        "label": str(card),
    }


@dataclass(frozen=True)
class OpponentChoice:
    key: str
    label: str
    policy: Policy


class GameSession:
    """Thread-safe adapter between the engine and an untrusted browser client."""

    def __init__(
        self,
        opponents: dict[str, OpponentChoice],
        *,
        players: int = 4,
        human_seat: int = 0,
        opponent_key: str = "counting",
        seed: int | None = None,
        dealer: int | None = None,
    ) -> None:
        self.opponents = opponents
        self.lock = threading.RLock()
        self.game: TwentyFiveGame
        self.human_seat = human_seat
        self.opponent_key = opponent_key
        self.policy = opponents[opponent_key].policy
        self.seed = 0
        self.rng = random.Random()
        self.initial_state: dict[str, Any] = {}
        self.actions: list[dict[str, int | str]] = []
        self.events: list[dict[str, Any]] = []
        self.new_game(players, human_seat, opponent_key, seed, dealer)

    def new_game(
        self,
        players: int,
        human_seat: int,
        opponent_key: str,
        seed: int | None,
        dealer: int | None,
    ) -> dict[str, Any]:
        with self.lock:
            if not rules.MIN_PLAYERS <= players <= rules.MAX_PLAYERS:
                raise ValueError(f"players must be between {rules.MIN_PLAYERS} and {rules.MAX_PLAYERS}")
            if not 0 <= human_seat < players:
                raise ValueError("human seat is outside the table")
            if opponent_key not in self.opponents:
                raise ValueError("unknown opponent")
            self.debug_seeded = seed is not None
            if seed is None:
                seed = random.SystemRandom().getrandbits(128)
            if dealer is None:
                dealer = seed % players
            if not 0 <= dealer < players:
                raise ValueError("dealer is outside the table")

            self.human_seat = human_seat
            self.opponent_key = opponent_key
            self.policy = self.opponents[opponent_key].policy
            self.seed = int(seed)
            self.rng = random.Random(self.seed ^ 0x25FACE)
            self.game = TwentyFiveGame(players, seed=self.seed, initial_dealer=dealer, debug=True)
            self.initial_state = self.game.to_dict()
            self.actions = []
            self.events = []
            self._event("round", f"Round {self.game.round_number} begins. {_SUIT_NAMES[int(self.game.trump)].title()} are trump.")
            self._advance_ai()
            return self.state()

    def _event(self, kind: str, message: str, **details: Any) -> None:
        self.events.append({"id": len(self.events) + 1, "kind": kind, "message": message, **details})

    def _record_action(self, seat: int, action: int, actor: str) -> Transition:
        phase = self.game.phase
        card = Card.from_index(action)
        previous_round = self.game.round_number
        self.actions.append({"seat": seat, "action": action, "actor": actor})
        transition = self.game.step(seat, action)

        if phase is Phase.ROB:
            if seat == self.human_seat:
                self._event("rob", f"You robbed and discarded {card} face down.", seat=seat)
            else:
                self._event("rob", f"Player {seat + 1} robbed and discarded one card face down.", seat=seat)
        else:
            who = "You" if seat == self.human_seat else f"Player {seat + 1}"
            self._event("play", f"{who} played {card}.", seat=seat, card=card_payload(card))

        if transition.trick_winner is not None:
            winner = transition.trick_winner
            who = "You" if winner == self.human_seat else f"Player {winner + 1}"
            self._event("trick", f"{who} won the trick.", seat=winner)
        if transition.game_winner is not None:
            winner = transition.game_winner
            who = "You win" if winner == self.human_seat else f"Player {winner + 1} wins"
            self._event("game", f"{who} the match with {self.game.scores[winner]} points!", seat=winner)
        elif self.game.round_number != previous_round:
            self._event("round", f"Round {self.game.round_number} begins. {_SUIT_NAMES[int(self.game.trump)].title()} are trump.")
        return transition

    def _advance_ai(self) -> None:
        actions = 0
        while not self.game.terminal and self.game.current_player != self.human_seat:
            if actions >= 1_000:
                raise RuntimeError("AI action safety limit exceeded")
            seat = self.game.current_player
            view = PlayerView.from_game(self.game, seat)
            action = self.policy.act(view, self.rng)
            if action not in {card.index for card in view.legal}:
                raise RuntimeError(f"{self.policy.name} selected an illegal action")
            self._record_action(seat, action, self.policy.name)
            actions += 1

    def act(self, action: int) -> dict[str, Any]:
        with self.lock:
            if self.game.terminal:
                raise InvalidAction("the match is already over")
            if self.game.current_player != self.human_seat:
                raise InvalidAction("it is not your turn")
            legal = {card.index for card in self.game.legal_cards(self.human_seat)}
            if action not in legal:
                raise InvalidAction("that card is not legal right now")
            self._record_action(self.human_seat, action, "human")
            self._advance_ai()
            return self.state()

    def state(self) -> dict[str, Any]:
        with self.lock:
            game = self.game
            legal = (
                {card.index for card in game.legal_cards(self.human_seat)}
                if not game.terminal and game.current_player == self.human_seat
                else set()
            )
            hand = sorted(game.hands[self.human_seat], key=lambda card: (int(card.suit), int(card.rank)))
            if game.terminal:
                prompt = "You won the match!" if game.winner == self.human_seat else f"Player {game.winner + 1} won the match."
            elif game.current_player != self.human_seat:
                prompt = f"Player {game.current_player + 1} is thinking…"
            elif game.phase is Phase.ROB:
                prompt = "You hold the ace of trump. Choose a non-trump card to discard face down."
            elif not game.current_trick:
                prompt = "Your lead. Choose a card."
            else:
                prompt = "Your turn. Choose one of the highlighted legal cards."

            return {
                "game": {
                    "players": game.num_players,
                    "round": game.round_number,
                    "phase": game.phase.value,
                    "dealer": game.dealer,
                    "leader": game.leader,
                    "current_player": game.current_player,
                    "trump": _SUIT_NAMES[int(game.trump)],
                    "trump_symbol": _SUIT_SYMBOLS[int(game.trump)],
                    "turned_card": card_payload(game.turned_card),
                    "scores": list(game.scores),
                    "round_tricks": list(game.round_tricks),
                    "current_trick": [
                        {"seat": seat, "card": card_payload(card)} for seat, card in game.current_trick
                    ],
                    "last_trick": [
                        {"seat": seat, "card": card_payload(card)}
                        for seat, card in (game.trick_history[-1] if game.trick_history else [])
                    ],
                    "robber": game.robber,
                    "robbed": game.rob_action is not None,
                    "terminal": game.terminal,
                    "winner": game.winner,
                    'debug_seeded': self.debug_seeded,
                    **({'seed': self.seed} if game.terminal or self.debug_seeded else {}),
                },
                "human": {
                    "seat": self.human_seat,
                    "hand": [dict(card_payload(card), legal=card.index in legal) for card in hand],
                    "legal_actions": sorted(legal),
                },
                "table": [
                    {
                        "seat": seat,
                        "name": "You" if seat == self.human_seat else f"Player {seat + 1}",
                        "is_human": seat == self.human_seat,
                        "hand_count": len(game.hands[seat]),
                        "score": game.scores[seat],
                        "tricks": game.round_tricks[seat],
                    }
                    for seat in range(game.num_players)
                ],
                "opponent": {"key": self.opponent_key, "label": self.opponents[self.opponent_key].label},
                "opponents": [
                    {"key": choice.key, "label": choice.label} for choice in self.opponents.values()
                ],
                "prompt": prompt,
                "events": self.events[-80:],
            }

    def replay_bytes(self) -> bytes:
        with self.lock:
            if not self.game.terminal:
                raise InvalidAction("the complete replay is available after the match ends")
            data = {
                "format_version": 1,
                "initial": self.initial_state,
                "actions": self.actions,
                "checkpoint": getattr(self.policy, "path", None),
                "opponent": self.opponent_key,
                "human_seat": self.human_seat,
            }
            if data["checkpoint"] is not None:
                data["checkpoint"] = str(data["checkpoint"])
            return (json.dumps(data, indent=2) + "\n").encode("utf-8")


def build_opponents(checkpoints: list[str]) -> dict[str, OpponentChoice]:
    choices = {
        name: OpponentChoice(name, name.replace("_", " ").title(), make_policy(name))
        for name in POLICIES
    }
    for specification in checkpoints:
        if "=" in specification:
            label, raw_path = specification.split("=", 1)
        else:
            raw_path = specification
            label = Path(raw_path).stem
        key = f"checkpoint:{label}"
        if key in choices:
            raise ValueError(f"duplicate opponent label {label!r}")
        path = Path(raw_path).expanduser().resolve()
        choices[key] = OpponentChoice(key, f"AI · {label}", FrozenCheckpointPolicy(path))
    return choices


def make_handler(session: GameSession) -> type[BaseHTTPRequestHandler]:
    mime_types = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml"}

    class GUIHandler(BaseHTTPRequestHandler):
        server_version = "TwentyFiveGUI/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 64_000:
                raise ValueError("request is too large")
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/state":
                self._json(session.state())
                return
            if path == "/api/replay":
                try:
                    body = session.replay_bytes()
                except InvalidAction as exc:
                    self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Disposition", 'attachment; filename="twenty-five-replay.json"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            asset = "index.html" if path in {"", "/"} else path.removeprefix("/")
            if asset not in {"index.html", "styles.css", "app.js"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = _asset_bytes(asset)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{mime_types[Path(asset).suffix]}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = self._body()
                if self.path == "/api/action":
                    self._json(session.act(int(body["action"])))
                elif self.path == "/api/new":
                    dealer_value = body.get("dealer")
                    seed_value = body.get("seed")
                    self._json(session.new_game(
                        int(body.get("players", 4)),
                        int(body.get("human_seat", 0)),
                        str(body.get("opponent", "counting")),
                        None if seed_value in (None, "") else int(seed_value),
                        None if dealer_value in (None, "", "random") else int(dealer_value),
                    ))
                else:
                    self._json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
            except (InvalidAction, KeyError, TypeError, ValueError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return GUIHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Twenty-Five in a local browser")
    parser.add_argument(
        "--checkpoint", action="append", default=[], metavar="[LABEL=]PATH",
        help="add a trained checkpoint to the opponent menu; repeat for several models",
    )
    parser.add_argument("--opponent", default="counting", help="initial built-in or checkpoint opponent key")
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--human-seat", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dealer", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    opponents = build_opponents(args.checkpoint)
    opponent_key = args.opponent
    if opponent_key not in opponents and args.checkpoint:
        opponent_key = next(key for key in opponents if key.startswith("checkpoint:"))
    session = GameSession(
        opponents, players=args.players, human_seat=args.human_seat,
        opponent_key=opponent_key, seed=args.seed, dealer=args.dealer,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    url = f"http://{args.host}:{args.port}"
    print(f"Twenty-Five table ready at {url}")
    print("Press Ctrl+C to stop the server.")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTable closed.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
