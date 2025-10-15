"""Simple test runner for the summarizer module."""
from summarizer import summarize


def test_basic():
    text = (
        "Today we discussed the Q3 roadmap. The engineering team will focus on performance improvements."
        " Marketing will prepare the new campaign. Next week we'll reconvene to review progress."
    )
    s = summarize(text, num_sentences=2)
    print('INPUT:', text)
    print('\nSUMMARY:', s)
    assert s, 'Summary should not be empty'
    assert len(s.split('.')) >= 1


def test_short_text():
    text = 'One sentence only.'
    s = summarize(text, num_sentences=3)
    assert s.strip() == text


if __name__ == '__main__':
    test_basic()
    test_short_text()
    print('All tests passed')
