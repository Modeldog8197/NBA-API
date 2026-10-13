import test from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';import os from 'node:os';import path from 'node:path';import {execFileSync} from 'node:child_process';
test('published HTML loads the archive adapter before both apps and all assets exist under the Pages subpath',()=>{
 const output=fs.mkdtempSync(path.join(os.tmpdir(),'nba-pages-check-'));
 try{
  execFileSync(process.execPath,['scripts/build-pages.mjs'],{env:{...process.env,PAGES_OUTPUT:output},stdio:'pipe'});
  for(const [file,app] of [['index.html','/NBA-API/app.js'],['shot-lab/index.html','/NBA-API/static/app.js']]){
   const html=fs.readFileSync(path.join(output,file),'utf8');
   const scripts=[...html.matchAll(/<script\b[^>]*src="([^"]+)"[^>]*>/g)].map(m=>m[1].split('?')[0]);
   assert.ok(scripts.includes('/NBA-API/pages-runtime.js'),file+' includes the runtime adapter');
   assert.ok(scripts.indexOf('/NBA-API/static/runtime-config.js')<scripts.indexOf('/NBA-API/pages-runtime.js'));
   assert.ok(scripts.indexOf('/NBA-API/pages-runtime.js')<scripts.indexOf(app),file+' loads adapter first');
   for(const asset of scripts){assert.ok(asset.startsWith('/NBA-API/'));assert.ok(fs.existsSync(path.join(output,asset.slice('/NBA-API/'.length))),asset)}
   for(const m of html.matchAll(/<link\b[^>]*href="([^"]+)"/g)){const url=m[1].split('?')[0];assert.ok(fs.existsSync(path.join(output,url.slice('/NBA-API/'.length))),url)}
  }
 }finally{fs.rmSync(output,{recursive:true,force:true})}
});
