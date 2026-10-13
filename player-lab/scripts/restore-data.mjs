import fs from 'node:fs';import path from 'node:path';import zlib from 'node:zlib';import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
// The preserved original project lives one level above this directory.
if(!fs.existsSync(path.join(root,'upstream')))fs.symlinkSync('..',path.join(root,'upstream'),'dir');
for(const folder of ['data','shot-data/processed','shot-data']){const dir=path.join(root,folder);if(!fs.existsSync(dir))continue;for(const file of fs.readdirSync(dir)){if(!file.endsWith('.json.gz')||/^20\d{2}\.json\.gz$/.test(file))continue;const dest=path.join(dir,file.slice(0,-3));if(!fs.existsSync(dest))fs.writeFileSync(dest,zlib.gunzipSync(fs.readFileSync(path.join(dir,file))))}}
