from typing import Literal
from MIAttack import MembershipInferenceWhiteBox, MembershipInference, MembershipInferenceBlackBox, MembershipInferenceLabelOnly, MembershipInferenceWhiteOld
import MIAttack
def get_attack(type:Literal['black','white','label','white_old'], **kwargs) -> MembershipInference:
    if(MIAttack.TYPE_BLACK == type):
        return MembershipInferenceBlackBox(**kwargs)
    elif(MIAttack.TYPE_WHITE == type):
        return MembershipInferenceWhiteBox(**kwargs)
    elif(MIAttack.TYPE_LABEL == type):
        return MembershipInferenceLabelOnly(**kwargs)
    elif(MIAttack.TYPE_WHITE_OLD == type):
        return MembershipInferenceWhiteOld(**kwargs)
    else:
        raise Exception('Non-implement attack')