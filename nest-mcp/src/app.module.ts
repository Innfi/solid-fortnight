import { Module } from '@nestjs/common';
import { McpModule } from '@rekog/mcp-nest';

import { AppController } from './app.controller';
import { AppService } from './app.service';

@Module({
  imports: [
    McpModule.forRoot({
      name: 'test-mcp-server',
      version: '1.0.0',
    }),
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
