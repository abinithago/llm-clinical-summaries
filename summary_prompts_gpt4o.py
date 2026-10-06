#!/usr/bin/env python3
"""
LLM prompting for clinical summary generation (AskDocs and MeDiSumQA)

This script generates prompts to create concise summaries of medical texts
from the MeDiSumQA and AskDocs datasets while preserving all clinically relevant information.
"""


class SummaryPrompter:
    """Generate summary prompts for MeDiSumQA and AskDocs datasets."""
    
    def __init__(self):
        # Define dataset-specific prompts
        self.prompt_templates = {
            'medisumqa': self._get_medisumqa_prompt,
            'askdocs': self._get_askdocs_prompt,
        }
        
    def _get_medisumqa_prompt(self) -> str:
        """Get MeDiSumQA dataset prompt with few-shot examples for discharge summary summarization."""
        return """You are a medical text summarization expert tasked with creating concise summaries of hospital discharge summaries while preserving ALL clinically relevant information.

Instructions:
- Create a concise summary that captures all essential clinical details
- Preserve ALL medically important information including:
  * Patient demographics (age, sex/gender when clinically relevant)
  * Chief complaint and reason for admission
  * History of present illness (HPI) - key symptoms, timeline, events
  * Past medical history (PMH) - relevant conditions
  * Physical examination findings
  * Laboratory and imaging results
  * Diagnoses
  * Procedures performed
  * Medications and treatments
  * Allergies
  * Discharge status and follow-up plans
- Remove only administrative/identifying information (names, unit numbers, specific dates when not medically relevant)
- Maintain medical accuracy and completeness
- Use clear, professional medical language
- Keep the summary concise but comprehensive - aim for 30-50% length reduction while retaining 100% of clinical information

Examples:

Example 1:
Original Discharge Summary:
Name:  ___                 Unit No:   ___
 
Admission Date:  ___              Discharge Date:   ___
 
Date of Birth:  ___             Sex:   M
 
Service: NEUROLOGY
 
Allergies: 
Aleve / Tapazole
 
Chief Complaint:
left eye mydriasis
 
History of Present Illness:
Patient is a 45 year old right handed man with PMH of reported
recurrent herpes zoster infection of the left eye and left side
of the head and hypothyroidism whom neurology has been consulted
because of fixed dilated left pupil.  

Patient thinks that his vision became more blurry than normal
this morning when he was walking his dog.  Patient did not
notice that there was something wrong with his eye when he was
getting ready for work.  Patient denies putting drops into his
eye or a foreign substance getting in his eye.  Patient does not
use nebulizers.  Patient walked into work and a co worker 
noticed that his left eye was red and dilated.

Physical Examination:
Neurological examination reveals fixed dilated left pupil. Visual acuity is decreased in the left eye. Otherwise normal neurological exam.

Assessment and Plan:
Fixed dilated left pupil, likely related to prior herpes zoster infection. Continue current medications. Follow-up with ophthalmology.

Summary:
45-year-old right-handed man with history of recurrent herpes zoster infection of the left eye and hypothyroidism, presenting with fixed dilated left pupil. Patient noted increased blurriness this morning while walking his dog, and a coworker observed red, dilated left eye. Patient denies eye drops or foreign substances. Neurological exam shows fixed dilated left pupil with decreased visual acuity in left eye; otherwise normal. Diagnosis: fixed dilated left pupil, likely related to prior herpes zoster infection. Plan: continue current medications, follow-up with ophthalmology. Allergies: Aleve/Tapazole.

Example 2:
Original Discharge Summary:
Name:  ___                    Unit No:   ___
 
Admission Date:  ___              Discharge Date:   ___
 
Date of Birth:  ___             Sex:   F
 
Service: SURGERY
 
Allergies: 
Penicillins
 
Chief Complaint:
Trauma:  roll-over MVC
LeFort 2 fracture

History of Present Illness:
This patient is a 32 year old female who complains of FACIAL FX. 
She is transferred from an outside hospital with a rollover MVC 
complicated by prolonged extrication. She was unrestrained. 
No known loss of consciousness. She had multiple facial fractures 
including LeFort 2 fracture and left wrist fracture.

Physical Examination:
Significant facial swelling and deformity. Left wrist deformity with pain on palpation. CT scan confirms LeFort 2 fracture.

Procedures:
Open reduction internal fixation of facial fractures. Left wrist reduction and casting.

Discharge Plan:
Stable for discharge. Follow-up with oral maxillofacial surgery and orthopedics in one week.

Summary:
32-year-old female transferred from outside hospital after rollover MVC with prolonged extrication. Unrestrained, no loss of consciousness. Multiple facial fractures including LeFort 2 fracture and left wrist fracture. Significant facial swelling and deformity; left wrist deformity with pain. CT confirms LeFort 2 fracture. Underwent open reduction internal fixation of facial fractures and left wrist reduction with casting. Discharged stable with follow-up scheduled for oral maxillofacial surgery and orthopedics in one week. Allergies: Penicillins.

Example 3:
Original Discharge Summary:
Date of Birth:  ___             Sex:   M
 
Service: MEDICINE
 
History of Present Illness:
The patient is a 68 year old man with PMH significant for HTN, 
uncontrolled DM2 (last A1C 9.2) complicated by peripheral neuropathy 
s/p right leg AKA and left toe amputation.

ROS: Was only able to answer a few questions since pt was so drowsy. 
Denies fever (states to feel warm), shortness of breath, chest pain, 
abdominal pain, nausea, vomiting, diarrhea, dysuria. He notes to have 
urinary frequency.

Physical Examination:
Vital signs stable. Drowsy but arousable. Lower extremity exam shows 
right AKA and left toe amputation site healing well. Neuropathy noted 
in remaining limbs.

Laboratory Results:
Glucose 450, A1C 9.2, creatinine elevated at 2.1.

Assessment:
Uncontrolled diabetes mellitus type 2 with hyperglycemia. Diabetic nephropathy. Rule out infection.

Plan:
Insulin therapy, blood glucose monitoring, continue current medications. Monitor renal function.

Summary:
68-year-old man with past medical history of hypertension and uncontrolled diabetes mellitus type 2 (last A1C 9.2) complicated by peripheral neuropathy, status post right leg above-knee amputation and left toe amputation. Patient was drowsy but arousable. Review of systems limited due to drowsiness; denies fever (but feels warm), shortness of breath, chest pain, abdominal pain, nausea, vomiting, diarrhea, dysuria; notes urinary frequency. Physical exam: stable vital signs, drowsy but arousable; right AKA and left toe amputation sites healing well; neuropathy in remaining limbs. Laboratory: glucose 450, A1C 9.2, creatinine elevated at 2.1. Assessment: uncontrolled diabetes mellitus type 2 with hyperglycemia, diabetic nephropathy, rule out infection. Plan: insulin therapy, blood glucose monitoring, continue current medications, monitor renal function.

Now, please create a concise summary of the following discharge summary that preserves ALL clinically relevant information:

{clinical_context}

Summary:"""
    
    def _get_askdocs_prompt(self) -> str:
        """Get AskDocs dataset prompt with few-shot examples for Reddit post summarization."""
        return """You are a medical text summarization expert tasked with creating concise summaries of Reddit r/AskDocs posts while preserving ALL clinically relevant information.

Instructions:
- Create a concise summary that captures all essential clinical details
- Preserve ALL medically important information including:
  * Patient demographics (age, gender/sex when provided)
  * Chief complaint and symptoms
  * Symptom timeline and progression
  * Relevant medical history
  * Medications and treatments tried
  * Physical characteristics mentioned (height, weight, etc.)
  * Lifestyle factors (smoking, drinking, etc.)
  * Specific questions or concerns
- Remove only conversational filler, excessive repetition, and non-medical personal details
- Maintain medical accuracy and completeness
- Use clear, professional medical language
- Keep the summary concise but comprehensive - aim for 30-50% length reduction while retaining 100% of clinical information

Examples:

Example 1:
Original Reddit Post:
My arms sometimes hurt when I sneeze? 28F, ex smoker, no drinking, 5'3", 200lbs. Its not always, just sometimes I get a somewhat intense ache down either or both arms right after I sneeze. Should I be worried?!

Summary:
28-year-old female, ex-smoker, non-drinker, 5'3", 200lbs. Reports intermittent arm pain occurring immediately after sneezing, affecting either or both arms with somewhat intense ache. Asks if this is concerning.

Example 2:
Original Reddit Post:
My friend's mom needs a liver. One of my best friend's mom is uninsured and has been diagnosed with cirrhosis of the liver. She's a sweet lady who doesn't drink or do drugs or anything like that. She's just a simple mother, wife, and homemaker. She's not expected to make it through the end of the year if she doesn't get a transplant, but being uninsured is a big obstacle. Does anybody have any information on a situation like this? Are there any organizations, charities, medical groups, or other institutions out there that can help work around the insurance issue? Or any insurance companies willing to take on a patient in immediate need of a liver transplant?

Summary:
Friend's mother diagnosed with cirrhosis of the liver, uninsured, not expected to survive end of year without liver transplant. Patient does not drink or use drugs. Seeking information about organizations, charities, medical groups, or insurance companies that can help with liver transplant for uninsured patient in immediate need.

Example 3:
Original Reddit Post:
Bump on toddler's neck for months. 3M, average height and weight, white, no medications or known medical issues. I noticed in at least early August that my 3.5 year old son has a bump on his neck where I believe a lymph node is located. It's been a few months now and it has not changed, if anything it has gotten slightly bigger. It is not present on the other side and is not tender to touch. Prior to noticing the bump, he was not recently sick that I can remember. I also can't remember if the bump was there before August but I think it's possible.

Summary:
3.5-year-old white male, average height and weight, no medications or known medical issues. Bump on neck (likely lymph node location) noticed since at least early August, present for several months. Bump has not changed or slightly increased in size. Unilateral (not present on other side), non-tender. No recent illness prior to noticing bump. Uncertain if bump was present before August.

Now, please create a concise summary of the following Reddit post that preserves ALL clinically relevant information:

{clinical_context}

Summary:"""
    
    def get_prompt(self, dataset_name: str, clinical_context: str) -> str:
        """Get the summary prompt for a dataset and clinical context."""
        # Normalize dataset name
        dataset_name = dataset_name.lower()
        
        prompt_func = self.prompt_templates.get(dataset_name)
        
        if prompt_func:
            prompt_template = prompt_func()
            return prompt_template.format(clinical_context=clinical_context)
        else:
            # Generic fallback prompt
            return f"""Create a concise summary of the following medical text that preserves ALL clinically relevant information:

{clinical_context}

Summary:"""


def main():
    """Example usage of the summary prompter."""
    prompter = SummaryPrompter()
    
    # Example MeDiSumQA discharge summary
    example_medisumqa = """Name:  ___                 Unit No:   ___
 
Admission Date:  ___              Discharge Date:   ___
 
Date of Birth:  ___             Sex:   F
 
Service: NEUROLOGY
 
Allergies: 
Aleve / Tapazole
 
Chief Complaint:
left eye mydriasis
 
History of Present Illness:
Patient is a ___ year old right handed woman with PMH of reported
recurrent herpes zoster infection of the left eye and left side
of the head and hypothyroidism whom neurology has been consulted
because of fixed dilated left pupil."""
    
    # Example AskDocs post
    example_askdocs = """My arms sometimes hurt when I sneeze? 28F, ex smoker, no drinking, 5'3", 200lbs. Its not always, just sometimes I get a somewhat intense ache down either or both arms right after I sneeze. Should I be worried?!"""
    
    print("=" * 80)
    print("MeDiSumQA Summary Prompt Example:")
    print("=" * 80)
    prompt1 = prompter.get_prompt('medisumqa', example_medisumqa)
    print(prompt1[:600] + "...")
    
    print("\n" + "=" * 80)
    print("AskDocs Summary Prompt Example:")
    print("=" * 80)
    prompt2 = prompter.get_prompt('askdocs', example_askdocs)
    print(prompt2[:600] + "...")


if __name__ == "__main__":
    main()

