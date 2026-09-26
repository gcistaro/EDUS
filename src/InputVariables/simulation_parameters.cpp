/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <unordered_set>
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>
#include "GlobalFunctions.hpp"
#include "simulation_parameters.hpp"
#include "Json/json_sirius.hpp"

/// Compose JSON dictionary with default parameters based on input schema.
/** Traverse the JSON schema and add nodes with default parameters to the output dictionary. The nodes without
 *  default parameters are ignored. Still, user has a possibility to add the missing nodes later by providing a
 *  corresponding input JSON dictionary. See compose_json() function. */
void
compose_default_json(nlohmann::json const& schema__, nlohmann::json& output__)
{
    for (auto it : schema__.items()) {
        auto key = it.key();
        /* this is a final node with the description of the data type */
        if (it.value().contains("type") && it.value()["type"] != "object") {
            /* check if default parameter is present */
            if (it.value().contains("default")) {
                output__[key] = it.value()["default"];
            }
        } else { /* otherwise continue to traverse the schema */
            if (!output__.contains(key)) {
                output__[key] = nlohmann::json{};
            }
            if (it.value().contains("properties")) {
                compose_default_json(it.value()["properties"], output__[key]);
            }
        }
    }
}


/// Append the input dictionary to the existing dictionary.
/** Use JSON schema to traverse the existing dictionary and add on top the values from the input dictionary. In this
 *  way we can add missing nodes which were not defined in the existing dictionary. */
void
compose_json(nlohmann::json const& schema__, nlohmann::json const& in__, nlohmann::json& inout__)
{
    compose_default_json(schema__, inout__);


    for (auto it : schema__.items()) {
        auto key = it.key();

        /* this is a final node with the description of the data type */
        /* for laser */
        if(it.value().contains("type") && it.value()["type"] == "array" && 
            it.value().contains("items") && it.value()["items"].contains("type") &&
            it.value()["items"]["type"] == "object" &&
            it.value()["items"].contains("properties")) {
             /* go though the object properties */
             if( in__.contains(key) ) {
                 auto aux = inout__[key]["properties"];
                 inout__[key] = std::vector<nlohmann::json>(in__[key].size());
                 for ( auto iobj=0; iobj < in__[key].size(); ++iobj ) {
                    std::stringstream iobj_str;
                    iobj_str << iobj;
                    inout__[key][iobj]=nlohmann::json();
                    compose_json(aux, 
                              in__[key][iobj], inout__[key][iobj]); 
                 }
             }
             else {
                 compose_json(inout__[key], 
                              nlohmann::json(), inout__[key]); 
             }
        }
        else if (it.value().contains("type") && it.value()["type"] != "object") {
            if (in__.contains(key)) {
                /* copy the new input */
                inout__[key] = in__[key];
            }
        }
    }
}

/// Number of single-character edits (insertions, deletions, substitutions) to go from a__ to b__
static int edit_distance(const std::string& a__, const std::string& b__)
{
    std::vector<int> previous(b__.size() + 1), current(b__.size() + 1);
    for (size_t j = 0; j <= b__.size(); ++j) {
        previous[j] = int(j);
    }
    for (size_t i = 1; i <= a__.size(); ++i) {
        current[0] = int(i);
        for (size_t j = 1; j <= b__.size(); ++j) {
            int substitution = previous[j-1] + ( std::tolower(a__[i-1]) == std::tolower(b__[j-1]) ? 0 : 1 );
            current[j] = std::min({previous[j] + 1, current[j-1] + 1, substitution});
        }
        std::swap(previous, current);
    }
    return previous[b__.size()];
}

/// Returns the key of the schema closest to key__, or an empty string if none is similar enough
static std::string closest_key(const std::string& key__, nlohmann::json const& properties__)
{
    std::string best;
    int best_distance = std::max(2, int(key__.size())/3) + 1;
    for (auto it : properties__.items()) {
        int distance = edit_distance(key__, it.key());
        if (distance < best_distance) {
            best_distance = distance;
            best = it.key();
        }
    }
    return best;
}

/// Finds the keys of the input that are not in the schema, which would be silently ignored
/// (e.g. misspelled parameters). properties__ are the properties of the schema at the level of in__.
/// Each entry of unknown__ is a message with the path of the key and, if any, the closest valid key.
static void find_unknown_keys(nlohmann::json const& properties__, nlohmann::json const& in__,
                              const std::string& path__, std::vector<std::string>& unknown__)
{
    if (!in__.is_object()) {
        return;
    }
    for (auto it : in__.items()) {
        auto& key = it.key();
        auto path = path__ + key;
        if (!properties__.contains(key)) {
            auto suggestion = closest_key(key, properties__);
            unknown__.push_back("'" + path + "'" + (suggestion.empty() ? "" : " (did you mean '" + suggestion + "'?)"));
            continue;
        }
        auto& schema = properties__[key];
        if (schema.contains("properties")) {
            /* object with its own parameters */
            find_unknown_keys(schema["properties"], it.value(), path + "/", unknown__);
        }
        else if (schema.contains("items") && schema["items"].is_object()) {
            auto& items = schema["items"];
            if (items.contains("properties") && it.value().is_array()) {
                /* array of objects, e.g. the lasers */
                for (size_t i = 0; i < it.value().size(); ++i) {
                    find_unknown_keys(items["properties"], it.value()[i], path + "[" + std::to_string(i) + "]/", unknown__);
                }
            }
            else if (!items.contains("type") && it.value().is_object()) {
                /* dictionary whose keys are listed in items, e.g. toprint */
                find_unknown_keys(items, it.value(), path + "/", unknown__);
            }
        }
    }
}

Config::Config()
{
    /* initialize JSON dictionary with default parameters */
    compose_default_json(EDUS::input_schema["properties"], this->dict_);

}

void
Config::import(nlohmann::json const& in__)
{
    /* the parameters that are not in the schema are ignored: warn the user */
    std::vector<std::string> unknown;
    find_unknown_keys(EDUS::input_schema["properties"], in__, "", unknown);
    if (!unknown.empty()) {
        output::title("INPUT WARNINGS");
        output::print("The following parameters are not recognized and are IGNORED:");
        for (auto& message : unknown) {
            output::print("  - ", message);
        }
    }

    /* overwrite the parameters by the values from the input dictionary */
    compose_json(EDUS::input_schema["properties"], in__, this->dict_);
}

void
Simulation_parameters::import(nlohmann::json const& dict__)
{
    cfg_.import(dict__);
}

void
Simulation_parameters::import(std::string const& str__)
{
    auto dict = read_json_from_file_or_string(str__);
    this->import(dict);
}

