from typing import Literal
from MIAttack import MembershipInferenceWhiteBox, MembershipInference, MembershipInferenceBlackBox
import MIAttack
def get_attack(type:Literal['black','white'], **kwargs) -> MembershipInference:
    if(MIAttack.TYPE_BLACK == type):
        return MembershipInferenceBlackBox(**kwargs)
    elif(MIAttack.TYPE_WHITE == type):
        return MembershipInferenceWhiteBox(**kwargs)
    else:
        raise Exception('Non-implement attack')