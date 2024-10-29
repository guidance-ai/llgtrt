use serde_json::Value;

pub fn remove_null(v: &mut serde_json::Value) {
    if let Some(map) = v.as_object_mut() {
        for (_, v) in map.iter_mut() {
            remove_null(v);
        }
        map.retain(|_, v| !v.is_null());
    }
    // remove empty arrays
    if let Some(arr) = v.as_array_mut() {
        arr.iter_mut().for_each(remove_null);
    }
}

pub fn json_merge(a: &mut Value, b: &Value) {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in b.iter() {
                json_merge(a.entry(k.clone()).or_insert(Value::Null), v);
            }
        }
        (a, b) => *a = b.clone(),
    }
}

fn write_indent(indent: usize, dst: &mut String) {
    for _ in 0..indent {
        dst.push(' ');
    }
}

fn write_comment(indent: usize, dst: &mut String, comment: Option<&str>) {
    if let Some(comment) = comment {
        for line in comment.lines() {
            write_indent(indent, dst);
            dst.push_str("/// ");
            dst.push_str(line);
            dst.push('\n');
        }
    }
}

const INDENT_LEVEL: usize = 2;

fn same_default(v: &Value, default: &Value) -> bool {
    match default {
        Value::Object(_) | Value::Array(_) => false,
        _ => v == default,
    }
}

fn json5_write(indent: usize, dst: &mut String, v: &Value, default: &Value, info: &Value) {
    match v {
        Value::Object(map) => {
            if map.is_empty() {
                dst.push_str("{}");
                return;
            }
            dst.push_str("{");
            for (k, v) in map.iter() {
                dst.push_str("\n");
                write_comment(indent + INDENT_LEVEL, dst, info[k]["#"].as_str());
                write_indent(indent + INDENT_LEVEL, dst);
                if same_default(v, &default[k]) {
                    dst.push_str("//");
                }
                dst.push_str(&serde_json::to_string_pretty(k).unwrap());
                dst.push_str(": ");
                json5_write(indent + INDENT_LEVEL, dst, v, &default[k], &info[k]);
                dst.push_str(",\n");
            }
            write_indent(indent, dst);
            dst.push('}');
        }
        Value::Array(arr) => {
            if arr.is_empty() {
                dst.push_str("[]");
                return;
            }
            dst.push_str("[\n");
            for v in arr.iter() {
                write_indent(indent + INDENT_LEVEL, dst);
                json5_write(indent + INDENT_LEVEL, dst, v, &Value::Null, info);
                dst.push_str(",\n");
            }
            write_indent(indent, dst);
            dst.push(']');
        }
        Value::String(_) | Value::Number(_) | Value::Bool(_) | Value::Null => {
            dst.push_str(&serde_json::to_string_pretty(v).unwrap());
        }
    }
}

pub fn json5_to_string(v: &Value, default: &Value, info: &Value) -> String {
    let mut dst = String::new();
    write_comment(0, &mut dst, info["#"].as_str());
    json5_write(0, &mut dst, v, default, info);
    dst
}
