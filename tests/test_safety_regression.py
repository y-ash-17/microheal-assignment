from app.safety import assess_safety


def test_hinglish_breathing_emergency():
    assert assess_safety("saans nahi aa rahi hai").level == 1
    assert assess_safety("mujhe saans lene me takleef ho rahi hai").level == 1
    assert assess_safety("saans fool rahi hai bohot").level == 1


def test_hinglish_chest_pain_and_bleeding():
    assert assess_safety("chhati me bahut tez dard ho raha hai").level == 1
    assert assess_safety("sine me tez dard hai").level == 1
    assert assess_safety("bohot tez bleeeding ruk nahi rahi hai, kya karu").level == 1


def test_hindi_devanagari_emergencies():
    assert assess_safety("सांस नहीं आ रही है").level == 1
    assert assess_safety("सीने में तेज दर्द है").level == 1
    assert assess_safety("मरीज बेहोश हो गया है").level == 1


def test_normal_request_is_not_an_emergency():
    assert assess_safety("book me cardiology").level is None
    assert assess_safety("Sunday wellness room hours kya hai").level is None
