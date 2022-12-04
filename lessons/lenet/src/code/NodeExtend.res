@module("fs")
external readFileStringSync: string => string = "readFileSync"

@module("fs")
external writeFileStringSync: (string, string) => unit = "writeFileSync"

@module("path")
external joinTwo: (string, string) => string = "join"

let getDirname = %raw(` () => { return __dirname} `)