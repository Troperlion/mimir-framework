"""Emit byte-exact shader text without large/raw C++ string literal pitfalls."""


def embedded_cpp_string(name, text, chunk_bytes=2048):
    if not 1 <= chunk_bytes <= 2048:
        raise ValueError('Expected chunks between 1 and 2048 bytes')
    data=text.encode('utf-8')
    def quote(chunk):
        escaped=[]
        for byte in chunk:
            if byte==34:escaped.append('\\"')
            elif byte==92:escaped.append('\\\\')
            elif 32<=byte<127:escaped.append(chr(byte))
            else:escaped.append('\\%03o'%byte)
        return '"'+''.join(escaped)+'"'
    lines=[f'inline const std::string {name}Storage=[] {{',
           '    std::string value;',f'    value.reserve({len(data)});']
    for start in range(0,len(data),chunk_bytes):
        chunk=data[start:start+chunk_bytes]
        lines.append(f'    value.append({quote(chunk)}, {len(chunk)});')
    lines.extend(['    return value;','}();',f'inline const char* {name}={name}Storage.c_str();'])
    return '\n'.join(lines)+'\n'
