"""Public-domain poem corpus for the tea-time skill.

All authors died before 1930; texts are public domain. The Bashō haiku is a
minimal literal rendering made for this repository (no copyrighted
translation is reproduced).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Poem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    title: str
    author: str
    lines: list[str]
    why_it_helps: str


POEM_CORPUS: dict[str, Poem] = {
    poem.key: poem
    for poem in [
        Poem(
            key="dickinson-tell-it-slant",
            title="Tell all the truth but tell it slant (1263)",
            author="Emily Dickinson",
            lines=[
                "Tell all the truth but tell it slant —",
                "Success in Circuit lies",
                "Too bright for our infirm Delight",
                "The Truth's superb surprise",
                "As Lightning to the Children eased",
                "With explanation kind",
                "The Truth must dazzle gradually",
                "Or every man be blind —",
            ],
            why_it_helps="a direct answer is not the only honest path to a result",
        ),
        Poem(
            key="whitman-learnd-astronomer",
            title="When I Heard the Learn'd Astronomer",
            author="Walt Whitman",
            lines=[
                "When I heard the learn'd astronomer,",
                "When the proofs, the figures, were ranged in columns before me,",
                "When I was shown the charts and diagrams, to add, divide, and measure them,",
                "When I sitting heard the astronomer where he lectured"
                " with much applause in the lecture-room,",
                "How soon unaccountable I became tired and sick,",
                "Till rising and gliding out I wander'd off by myself,",
                "In the mystical moist night-air, and from time to time,",
                "Look'd up in perfect silence at the stars.",
            ],
            why_it_helps="leave the charts for a moment and look at the raw phenomenon",
        ),
        Poem(
            key="blake-grain-of-sand",
            title="Auguries of Innocence (opening)",
            author="William Blake",
            lines=[
                "To see a World in a Grain of Sand",
                "And a Heaven in a Wild Flower,",
                "Hold Infinity in the palm of your hand",
                "And Eternity in an hour.",
            ],
            why_it_helps="one boring data point, examined closely enough, contains the whole story",
        ),
        Poem(
            key="basho-old-pond",
            title="Old pond (furu ike ya)",
            author="Matsuo Basho",
            lines=[
                "old pond —",
                "a frog leaps in,",
                "the sound of water",
            ],
            why_it_helps="the signal is the brief disturbance, not the still background",
        ),
    ]
}
