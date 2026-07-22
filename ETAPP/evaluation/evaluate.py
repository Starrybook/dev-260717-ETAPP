import os
import json
import argparse
import logging
from evaluation.evaluate_prompt import *
import re
import time
from pathlib import Path
from evaluation.common import (
    build_evaluation_parser,
    finalize_evaluation_args,
    load_results,
    make_sample_key,
    request_evaluation,
    score_with_retries,
    is_successful,
    write_results,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

transfor_dict = {'get_current_health_and_mood_status': 'health', 
                 'get_recent_health_and_mood_summary': 'health', 
                 'get_user_recent_workout_records': 'health', 
                 'add_event_in_calendar': 'calendar', 
                 'view_today_events_in_calendar': 'calendar', 
                 'view_events_in_calendar_by_providing_time_range': 'calendar', 
                 'delete_event_in_calendar': 'calendar', 
                 'add_alarm': 'calendar', 
                 'remove_alarm': 'calendar', 
                 'view_today_alarms': 'calendar', 
                 'add_product_to_cart': 'shopping', 
                 'remove_product_from_cart': 'shopping', 
                 'purchase_product_in_shopping_manager': 'shopping', 
                 'search_products_in_shopping_manager': 'shopping', 
                 'get_status_information_of_purchased_products': 'shopping', 
                 'view_cart_in_shopping_manager': 'shopping', 
                 'send_email': 'email', 
                 'get_today_emails_until_now': 'email', 
                 'search_email_by_sender_and_receiver': 'email', 
                 'search_email_by_content': 'email', 
                 'delete_email': 'email', 
                 'search_news_by_category': 'web_browsing', 
                 'search_heat_news': 'web_browsing', 
                 'search_from_wikipedia': 'web_browsing',
                 'play_music': 'music', 
                 'search_music_by_name': 'music', 
                 'get_music_list_in_favorites': 'music', 
                 'find_accommodations': 'navigation', 
                 'find_attractions': 'navigation', 
                 'find_restaurants': 'navigation', 
                 'find_flight': 'navigation', 
                 'set_temperature_and_humidity_in_home': 'smart_home_devices', 
                 'get_home_temperature_and_humidity': 'smart_home_devices', 
                 'control_light_in_home': 'smart_home_devices', 
                 'get_lighting_status_in_home': 'smart_home_devices', 
                 'control_curtains_in_home': 'smart_home_devices', 
                 'control_bathtub_in_home': 'smart_home_devices', 
                 'boil_water_in_home': 'smart_home_devices',
                 'get_today_weather': '',
                 'get_future_weather': '',
                 'get_weather_for_current_hour': ''}


def get_instruction_id(_user_index, result_index):
    return result_index

def get_score(model_name, base_url=None, message=None, api_key=None,
              max_tokens=8192, temperature=0.0, request_timeout=None):
    evaluation_results, _ = request_evaluation(
        model_name=model_name,
        messages=message or [],
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        request_timeout=request_timeout,
    )
    return evaluation_results

        
def evaluate_personalization_and_proactivity(instruction, profile, profile_dict, result, evaluation_models, plan=False, model="gpt-4o", evaluation_config=None):
    timestamp = result['timestamp']

    
    profile = json.dumps(profile)
    query = result['query']
    output = []
    if plan:
        preoutput = result['output'][4:]
    else:
        preoutput = result['output'][2:]
    # output = result['output'][2:]
    for item_data in preoutput:
        if item_data["role"] == "assistant" and "tool_calls" not in item_data:
            output.append(item_data)
        elif item_data["role"] == "assistant" and type(item_data["tool_calls"]) == dict:
            assert item_data["tool_calls"]["name"] == "finish"
            output.append({"role": "assistant", "content": item_data["tool_calls"]["arguments"], "tool_calls": None})
        elif item_data["role"] == "assistant" and item_data["tool_calls"] == None:
            output.append(item_data)
        elif item_data["role"] == "assistant" and len(item_data["tool_calls"]) == 0:
            output.append({"role": "assistant", "content": item_data["content"], "tool_calls": None})
        elif item_data["role"] == "assistant" and item_data["tool_calls"][0]["name"] == "finish":
            output.append({"role": "assistant", "content": item_data["content"] + "\n" + item_data["tool_calls"][0]["arguments"], "tool_calls": None})
        else:
            output.append(item_data)
    
    
    output = [str(item) for item in output]
    output = "\n".join(output)

    keypoint_for_personal = result['keypoint for personal']
    keypoint_for_proactive = result['keypoint for proactive']
    personal = {}
    proactive = {}
    for item in keypoint_for_personal:
        personal[item] = {"analysis": "<You need to analyze whether the key point is met in the model's output. If it is met, clearly indicate which tools or parameters were used, or which specific aspect of the final answer satisfied the key point. If it is not met, analyze whether the key point does not need to be considered due to objective reasons (for example, there is no need to consider bringing an umbrella on a sunny day). >", "score": "<`0` if not satisfied, `1` if part satisfied, `2` if satisfied>"}
        
    for item in keypoint_for_proactive:
        proactive[item] = {"analysis": "<You need to analyze whether the key point is met in the model's output. If it is met, clearly indicate which tools or parameters were used, or which specific aspect of the final answer satisfied the key point. If it is not met, analyze whether the key point does not need to be considered due to objective reasons (for example, there is no need to consider bringing an umbrella on a sunny day).>", "score": "<`0` if not satisfied, `1` if part satisfied, `2` if satisfied>"}
        
    
    output_format = {
        "Procedure":{
            "Keypoints for Procedure": {
                "Completeness": {"analysis": "<Analyze whether the response fully addresses the query>", "score": "<`0` if not satisfied, `1` if part satisfied, `2` if satisfied>"},
                "Avoid Unneccessary Action": {"analysis": "<Evaluate if any redundant or irrelevant actions are present>", "score": "<`0` if not satisfied, `1` if part satisfied, `2` if satisfied>"},
                "Call the tool accurate": {"analysis": "<Check if each tool call is in the correct format and necessary>", "score": "<`0` if not satisfied, `1` if part satisfied, `2` if satisfied>"},
                "Summary the query clearly and comperhensive": {"analysis": "<Assess if the query has been summarized accurately and fully>", "score": "<`0` if not satisfied, `1` if part satisfied, `2` if satisfied>"}
            },
            "Final Assessment": {"anlysis": "Analyze the score of Procedure", "score": "<A score from 0 to 5 based on the Procedure criteria>"}
        },
        "Personalization": {
            "Keypoints for Personalization": personal,
            "Final Assessment": {"analysis": "Analyze the score of Personalization", "score": "<A score from 0 to 5 based on the Personalization criteria>"}
        },
        "Proactivity": {
            "Keypoints for Proactivity": proactive,
            "Final Assessment": {"Analysis": "Analyze the score of the Proative", "score": "<A score from 0 to 5 based on the Proactive criteria>"}
        }
    }
    output_format = json.dumps(output_format, indent=4)

    message = [{"role": "user", "content": EVALUATE_PERSONALIZATION_AND_PROACTIVITY.format(profile=profile + "\n" + str(profile_dict), query=query, personal=personal, proactive=proactive, output_format=output_format, keypoint_for_personal=keypoint_for_personal, keypoint_for_proactive=keypoint_for_proactive, output=str(output))}]
    # print(message[0]["content"])
    logging.info(message[0]["content"])
    scores = {}
    evaluation_config = evaluation_config or {}
    for model_name in evaluation_models:
        scores[model_name] = score_with_retries(
            model_name=model_name,
            messages=message,
            base_url=evaluation_config.get("evaluation_base_url"),
            api_key=evaluation_config.get("evaluation_api_key"),
            max_tokens=evaluation_config.get("evaluation_max_tokens", 8192),
            temperature=evaluation_config.get("evaluation_temperature", 0.0),
            request_timeout=evaluation_config.get("evaluation_request_timeout"),
        )
        logging.info("%s: %s", model_name, scores[model_name])
    
    # breakpoint()
    return [result], scores


def main(params):
    with open(params['profile_file'], 'r') as profile_f:
        profiles = json.load(profile_f)
    evaluation_models = [params.get("evaluation_model", "gpt-4o-2024-11-20")]
    with open(params["instruction_file"], 'r') as f:
        all_instruction = json.load(f)  
    plan = True if "e-react" in params['result_file'] else False
    all_evaluate_result = load_results(params["evaluate_result_file"])
    completed_keys = {
        item.get("sample_key") for item in all_evaluate_result if is_successful(item)
    }
    setting = params.get("setting") or Path(params["result_file"]).name
    for i, (person, profile) in enumerate(profiles.items()):
        print(i)
        person_name = person.replace(" ", "_")
        file_name = f"{person_name}_instruction.json"
        result_file = os.path.join(params['result_file'], file_name)

        with open(result_file, 'r') as f:
            model_result = json.load(f)

        with open(os.path.join(params["concrete_profile_dir"], f"profile_{person_name}.json"), "r") as f:
            profile_dict = json.load(f)
        
        # for idx, result in enumerate(model_result[a[i]: a[i]+1]):
        #     instruction = all_instruction[a[i]]
        for idx, result in enumerate(model_result):
            instruction_id = get_instruction_id(i, idx)
            sample_key = make_sample_key(setting, evaluation_models[0], person_name, instruction_id)
            if sample_key in completed_keys:
                continue
            print(f"processing {person}, instruction id {instruction_id}")
            instruction = all_instruction[instruction_id]
            assert result["query"] == instruction["query"]
            preferences = []
            available_tool_names = result["available_tools_name"]
            already_add = []
            for tool_name in available_tool_names:
                if transfor_dict[tool_name] != "" and transfor_dict[tool_name] not in already_add:
                    already_add.append(transfor_dict[tool_name])
                    preferences.append(profile_dict[transfor_dict[tool_name]])
            used_tools = []
            for item in result["output"]:
                if item["role"] == "assistant":
                    if "tool_calls" not in item:
                        continue
                    if type(item["tool_calls"]) == list:
                        for item_tool in item["tool_calls"]:
                            # print(item_tool)
                            # print(list(transfor_dict.keys()))
                            if item_tool["name"] in list(transfor_dict.keys()):
                                used_tools.append(item_tool["name"])
                    elif type(item["tool_calls"]) == dict:
                        item_tool = item["tool_calls"]
                        assert item_tool["name"] == "finish"
                        if item_tool["name"] in list(transfor_dict.keys()):
                            used_tools.append(item_tool["name"])
                    else:
                        continue
            
            for tool_name in used_tools:
                if transfor_dict[tool_name] != "" and transfor_dict[tool_name] not in already_add:
                    already_add.append(transfor_dict[tool_name])
                    preferences.append(profile_dict[transfor_dict[tool_name]])
            evaluate_result, score = evaluate_personalization_and_proactivity(
                instruction, profile=profile, profile_dict=preferences, result=result,
                evaluation_models=evaluation_models, plan=plan, evaluation_config=params,
            )
            all_evaluate_result = [item for item in all_evaluate_result if item.get("sample_key") != sample_key]
            all_evaluate_result.append({
                "sample_key": sample_key,
                "setting": setting,
                "user": person_name,
                "instruction_id": instruction_id,
                "query": params["result_file"] + "_" + person_name + "_" + result["query"],
                "evaluation_result": score,
            })
            write_results(all_evaluate_result, params["evaluate_result_file"], params["summary_file"])

        
        summary = write_results(all_evaluate_result, params["evaluate_result_file"], params["summary_file"])
    summary = write_results(all_evaluate_result, params["evaluate_result_file"], params["summary_file"])
    print(summary)
    logging.info(summary)


    return


if __name__ == "__main__":
    parser = build_evaluation_parser()
    args = finalize_evaluation_args(parser.parse_args())
        
    params = args.__dict__
    c_time = time.time()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=args.logging_dir,
        filemode='w' 
    )
    
    main(params)
