# Exercise the actual config-driven CLI, including task selection and failures.
set(test_dir "${CMAKE_CURRENT_BINARY_DIR}/js conf smoke")
file(MAKE_DIRECTORY "${test_dir}")
set(script "${test_dir}/check.mjs")
file(WRITE "${script}" [=[
#!/usr/bin/env node
import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
if (fileURLToPath(import.meta.url) !== path.resolve(arg[0]) || process.argv[1] !== path.resolve(arg[0]))
  throw new Error('original script identity lost');
if (fs.readdirSync(path.dirname(arg[0])).some(name => name.startsWith('.mimir-js-')))
  throw new Error('bridge wrote into the source directory');
if (CONF.marker !== 42 || !CONF_PATH.endsWith('/config.json') ||
    !CONF_DIR.endsWith('/js conf smoke') || arg[1] !== 'é漢字 | spaces')
  throw new Error('JS configuration or argument injection failed');
if (Model !== Mimir.Model || model !== Model) throw new Error('aliases');
console.log('JS_CONF_OK');
]=])
# Escape Windows-style separators in JSON (the bridge currently targets POSIX).
string(REPLACE "\\" "/" script "${script}")
set(block "{\"scripts\":[{\"script\":\"${script}\",\"args\":[\"é漢字 | spaces\"]}]}")
foreach(mode root nested task)
    set(options)
    if(mode STREQUAL root)
        set(config "{\"marker\":42,\"js\":${block}}")
    elseif(mode STREQUAL nested)
        set(config "{\"marker\":42,\"run\":{\"js\":${block}}}")
    else()
        set(config "{\"marker\":42,\"tasks\":{\"check\":{\"js\":${block}}}}")
        set(options --run check)
    endif()
    file(WRITE "${test_dir}/config.json" "${config}")
    execute_process(COMMAND "${MIMIR_BIN}" --conf "${test_dir}/config.json" ${options}
        RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT status EQUAL 0 OR NOT output MATCHES "JS_CONF_OK")
        message(FATAL_ERROR "JS ${mode} config failed: ${status}\n${output}\n${error}")
    endif()
endforeach()
file(WRITE "${test_dir}/args.mjs" [=[
if (arg[1] !== '--literal' || arg[2] !== 'é漢字 | spaces' || arg.length !== 3)
  throw new Error('CLI argument separator lost');
]=])
execute_process(COMMAND "${MIMIR_BIN}" --js "${test_dir}/args.mjs" -- --literal "é漢字 | spaces"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "JS argument separator failed: ${status}\n${output}\n${error}")
endif()
file(WRITE "${script}" "throw new Error('intentional failure');\n")
execute_process(COMMAND "${MIMIR_BIN}" --js "${script}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(status EQUAL 0)
    message(FATAL_ERROR "Uncaught JS error returned success")
endif()
file(GLOB leftovers "${test_dir}/.mimir-js-*")
if(leftovers)
    message(FATAL_ERROR "Temporary JS modules leaked: ${leftovers}")
endif()
file(REMOVE_RECURSE "${test_dir}")
