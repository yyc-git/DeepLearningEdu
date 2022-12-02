let throwErr = %raw(` (err) => { throw err; } `)

let buildErr = %raw(` (message) => { return new Error(message); } `)

let notImplement = () => {
  throwErr({j`not implement`})
}
